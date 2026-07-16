import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
warnings.filterwarnings('ignore')

# Optimized configuration for better accuracy without overfitting
MAX_SEQ_LEN = 40  # Reduced from 50 to prevent overfitting on long sequences
BATCH_SIZE = 64   # Increased for more stable gradients
EPOCHS = 100      # Reduced from 150
EMBED_DIM = 96    # Slightly reduced to prevent overfitting
HIDDEN_DIM = 192  # Balanced capacity
LEARNING_RATE = 0.001  # Slightly higher for better convergence
DROPOUT = 0.35    # Slightly reduced for better learning
WEIGHT_DECAY = 0.0005  # Increased regularization
SAVE_PATH = r"C:\Users\Faster\Downloads\Autonomous\balanced_enhanced_classifier.pt"

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

class BalancedCommandDataset(Dataset):
    def __init__(self, sequences, labels, max_len=MAX_SEQ_LEN, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
    
    def augment_sequence(self, seq):
        """Light data augmentation to improve generalization"""
        if not self.augment or random.random() > 0.3:
            return seq
        
        seq = seq.copy()
        # Randomly drop some tokens (simulates missing data)
        if len(seq) > 3 and random.random() < 0.15:
            drop_idx = random.randint(1, len(seq) - 2)
            seq.pop(drop_idx)
        
        # Randomly shuffle adjacent non-critical tokens (simulates reordering)
        if len(seq) > 4 and random.random() < 0.1:
            i = random.randint(1, len(seq) - 3)
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
        
        return seq
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply augmentation
        if self.augment:
            seq = self.augment_sequence(seq)
        
        # Truncate or pad sequence
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        actual_length = min(len(seq), self.max_len)
        
        return {
            'sequence': torch.tensor(seq, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': actual_length
        }

def balanced_collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = [item['length'] for item in batch]
    
    max_len = max(len(seq) for seq in sequences)
    max_len = min(max_len, MAX_SEQ_LEN)
    
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded_seqs[i, :length] = seq[:length]
    
    return {
        'sequences': padded_seqs,
        'labels': labels,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

class BalancedClassifier(nn.Module):
    """Balanced model architecture focused on accuracy without overfitting"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        
        # Embedding with controlled capacity
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout * 0.5)  # Lower dropout for embeddings
        
        # Single bidirectional LSTM (simpler than multi-layer)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0,  # No internal dropout to prevent underfitting
            num_layers=1
        )
        
        # Simplified attention mechanism
        self.attention_dim = hidden_dim * 2
        self.attention = nn.Linear(self.attention_dim, 1, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Feature processing
        self.feature_dim = hidden_dim * 4  # max + mean pooling from bidirectional LSTM
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # Simplified but effective classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # Reduced dropout in middle layers
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Even less dropout before final layer
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Improved weight initialization"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Forget gate bias
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
            elif 'embedding' in name:
                nn.init.normal_(param.data, mean=0, std=0.1)
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
    def forward(self, x, lengths=None):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Create mask for variable length sequences
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)  # (batch, seq, 1)
        else:
            mask = torch.ones(batch_size, seq_len, 1, device=x.device)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq, 1)
        attention_scores = attention_scores.masked_fill(mask.squeeze(-1).unsqueeze(-1) == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Weighted attention output
        attended_output = torch.sum(lstm_out * attention_weights * mask, dim=1)  # (batch, hidden*2)
        
        # Global pooling
        masked_lstm = lstm_out * mask
        max_pooled = torch.max(masked_lstm, dim=1)[0]  # (batch, hidden*2)
        
        # Combine features
        combined_features = torch.cat([attended_output, max_pooled], dim=1)  # (batch, hidden*4)
        combined_features = self.feature_norm(combined_features)
        
        # Classification
        output = self.classifier(combined_features)
        return output

class SmoothLabelLoss(nn.Module):
    """Label smoothing + class weighting for better generalization"""
    def __init__(self, num_classes, smoothing=0.1, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.register_buffer('class_weights', class_weights)
        
    def forward(self, pred, target):
        batch_size = pred.size(0)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        log_prob = F.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_prob, dim=1)
        
        if self.class_weights is not None:
            weight = self.class_weights[target]
            loss = loss * weight
            
        return loss.mean()

def plot_enhanced_training_history(train_losses, val_losses, train_accs, val_accs, 
                                 train_f1s, val_f1s, save_path):
    """Enhanced training history visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(train_accs, label='Training Accuracy', color='blue', alpha=0.7)
    axes[0, 1].plot(val_accs, label='Validation Accuracy', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[1, 0].plot(train_f1s, label='Training F1', color='blue', alpha=0.7)
    axes[1, 0].plot(val_f1s, label='Validation F1', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    axes[1, 1].text(0.5, 0.5, 'Model Architecture:\n\n' + 
                   f'Vocabulary: {len(train_losses)} epochs\n' +
                   f'Embedding: {EMBED_DIM}D\n' +
                   f'Hidden: {HIDDEN_DIM}D\n' +
                   f'Dropout: {DROPOUT}\n' +
                   f'Batch Size: {BATCH_SIZE}',
                   transform=axes[1, 1].transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Training Configuration')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.pt', '_enhanced_training_history.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def balanced_evaluate(model, val_loader, criterion, device, label_encoder, verbose=True):
    """Balanced evaluation with comprehensive metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            seqs = batch['sequences'].to(device)
            labs = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labs)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    if verbose:
        print(f"\nValidation Results:")
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"F1-Macro: {f1_macro:.4f}, F1-Weighted: {f1_weighted:.4f}")
        
        print("\nPer-Class Performance:")
        target_names = label_encoder.classes_
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels, np.array(all_probs)

def balanced_cross_validate(sequences, labels, vocab_size, num_classes, device, label_encoder, class_weights=None):
    """Balanced cross-validation with proper regularization"""
    print("\n" + "="*60)
    print("BALANCED 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'loss': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\nFold {fold + 1}/5:")
        
        X_train = [sequences[i] for i in train_idx]
        X_val = [sequences[i] for i in val_idx]
        y_train = [labels[i] for i in train_idx]
        y_val = [labels[i] for i in val_idx]
        
        # Create datasets with augmentation only for training
        train_dataset = BalancedCommandDataset(X_train, y_train, augment=True)
        val_dataset = BalancedCommandDataset(X_val, y_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=balanced_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              collate_fn=balanced_collate_fn)
        
        # Create model
        model = BalancedClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)
        
        # Balanced loss function
        class_weight_tensor = torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
        criterion = SmoothLabelLoss(num_classes, smoothing=0.1, class_weights=class_weight_tensor)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Training loop for CV
        best_f1 = 0
        patience_counter = 0
        for epoch in range(40):  # Reasonable epochs for CV
            model.train()
            
            for batch in train_loader:
                seqs = batch['sequences'].to(device)
                labs = batch['labels'].to(device)
                lengths = batch['lengths'].to(device)
                
                optimizer.zero_grad()
                outputs = model(seqs, lengths)
                loss = criterion(outputs, labs)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            # Quick validation
            val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _, _ = balanced_evaluate(
                model, val_loader, criterion, device, label_encoder, verbose=False
            )
            
            if val_f1_macro > best_f1:
                best_f1 = val_f1_macro
                best_acc = val_acc
                best_loss = val_loss
                best_f1_weighted = val_f1_weighted
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Early stopping for CV
                    break
        
        cv_results['accuracy'].append(best_acc)
        cv_results['f1_macro'].append(best_f1)
        cv_results['f1_weighted'].append(best_f1_weighted)
        cv_results['loss'].append(best_loss)
        
        print(f"  Best - Acc: {best_acc:.4f}, F1: {best_f1:.4f}")
    
    print(f"\nCross-validation Summary:")
    for metric, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    return cv_results

def main():
    # Load data
    data_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_smart.npz"
    label_enc_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_smart_label_encoder.pkl"
    vocab_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_smart_vocab.pkl"
    
    print("Loading preprocessed data...")
    data_npz = np.load(data_path, allow_pickle=True)
    sequences_padded = data_npz['sequences']
    labels = data_npz['labels']
    sequence_lengths = data_npz.get('sequence_lengths', None)
    
    # Convert to proper sequences
    if sequence_lengths is not None:
        sequences = []
        for i, length in enumerate(sequence_lengths):
            sequences.append(sequences_padded[i][:length].tolist())
    else:
        sequences = []
        for seq in sequences_padded:
            seq_list = seq.tolist()
            last_nonzero = len(seq_list) - 1
            while last_nonzero >= 0 and seq_list[last_nonzero] == 0:
                last_nonzero -= 1
            sequences.append(seq_list[:last_nonzero + 1])
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Load vocabulary and label encoder
    with open(label_enc_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Using device: {device}")
    
    # Calculate balanced class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(f"Class weights: {dict(zip(label_encoder.classes_, class_weights))}")
    
    # Cross-validation
    cv_results = balanced_cross_validate(sequences, labels, vocab_size, num_classes, 
                                       device, label_encoder, class_weights)
    
    # Final training
    print(f"\n{'='*80}")
    print("BALANCED FINAL TRAINING")
    print(f"{'='*80}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create datasets
    train_dataset = BalancedCommandDataset(X_train, y_train, augment=True)
    val_dataset = BalancedCommandDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=balanced_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          collate_fn=balanced_collate_fn, num_workers=0)
    
    # Create balanced model
    model = BalancedClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    class_weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = SmoothLabelLoss(num_classes, smoothing=0.1, class_weights=class_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    # Training tracking
    best_f1_macro = 0
    patience = 20
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    print("Starting balanced training...")
    print("-" * 80)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        all_train_preds = []
        all_train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS}", leave=False)
        for batch in train_pbar:
            seqs = batch['sequences'].to(device)
            labs = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labs)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item() * seqs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labs).sum().item()
            total_train += labs.size(0)
            
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labs.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = correct_train / total_train
        avg_train_loss = total_train_loss / total_train
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        
        # Validation
        val_loss, val_acc, val_f1_macro, val_f1_weighted, val_preds, val_labels, val_probs = balanced_evaluate(
            model, val_loader, criterion, device, label_encoder, 
            verbose=(epoch % 25 == 0 or epoch == EPOCHS-1)
        )
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1_macro)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"E{epoch+1:3d} | Train: L{avg_train_loss:.4f} A{train_acc:.4f} F{train_f1:.4f} | "
              f"Val: L{val_loss:.4f} A{val_acc:.4f} F{val_f1_macro:.4f} | LR{current_lr:.6f}")
        
        # Model saving and early stopping
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            patience_counter = 0
            
            # Save best model
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_classes': num_classes,
                'dropout': DROPOUT,
                'best_val_f1': best_f1_macro,
                'best_val_acc': val_acc,
                'epoch': epoch,
                'cv_results': cv_results,
                'class_weights': class_weights,
                'label_encoder': label_encoder,
                'vocab': vocab,
                'training_config': {
                    'max_seq_len': MAX_SEQ_LEN,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY
                }
            }
            torch.save(checkpoint, SAVE_PATH)
            print(f"    ✓ New best F1: {best_f1_macro:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Final results
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Cross-validation F1: {np.mean(cv_results['f1_macro']):.4f} ± {np.std(cv_results['f1_macro']):.4f}")
    print(f"Best validation F1: {best_f1_macro:.4f}")
    print(f"Gap (CV vs Final): {abs(np.mean(cv_results['f1_macro']) - best_f1_macro):.4f}")
    
    if abs(np.mean(cv_results['f1_macro']) - best_f1_macro) < 0.02:
        print("✓ Good generalization - CV and final results are consistent")
    else:
        print("⚠ Possible overfitting - large gap between CV and final results")
    
    # Create visualizations
    plot_enhanced_training_history(train_losses, val_losses, train_accs, val_accs, 
                                 train_f1s, val_f1s, SAVE_PATH)
    
    print(f"\nFiles created:")
    print(f"  - Model: {SAVE_PATH}")
    print(f"  - Training history: {SAVE_PATH.replace('.pt', '_enhanced_training_history.png')}")

if __name__ == '__main__':
    main()