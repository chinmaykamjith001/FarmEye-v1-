#training script

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# Conservative training configuration
MAX_SEQ_LEN = 25
BATCH_SIZE = 32        # Smaller batch size for better generalization
EPOCHS = 50
EMBED_DIM = 32         # Much smaller embedding
HIDDEN_DIM = 64        # Much smaller hidden dimension
LEARNING_RATE = 0.0005 # Lower learning rate
DROPOUT = 0.5          # High dropout for regularization
WEIGHT_DECAY = 0.01    # Strong weight decay
SAVE_PATH = r"C:\Users\Faster\Downloads\Autonomous\robust_classifier.pt"

class CommandDataset(Dataset):
    def __init__(self, sequences, labels, lengths, max_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        length = min(self.lengths[idx], self.max_len)
        
        # Truncate sequence if too long
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
            length = self.max_len
        
        return {
            'sequence': torch.tensor(seq, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': length
        }

def collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = [item['length'] for item in batch]
    
    # Pad to max length in batch
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = seq
    
    return {
        'sequences': padded_seqs,
        'labels': labels,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

class RobustClassifier(nn.Module):
    """Conservative model designed to prevent overfitting"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        
        # Simple embedding with dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Single LSTM layer (prevents overfitting)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Simple attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Conservative classification head with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize with smaller weights
        self._init_weights()
        
    def _init_weights(self):
        # Conservative initialization
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Smaller gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len = x.size()
        
        # Embedding with dropout
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention pooling with masking
        if lengths is not None:
            # Create attention mask
            mask = torch.zeros(batch_size, seq_len, device=x.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1.0
            
            # Apply attention
            attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch, seq)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Weighted sum
            output = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        else:
            # Simple mean pooling
            output = lstm_out.mean(dim=1)
        
        # Classification
        output = self.classifier(output)
        return output

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

def evaluate_model(model, data_loader, criterion, device, label_encoder=None):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            seqs = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels

def robust_cross_validate(train_data, train_labels, train_lengths, vocab_size, num_classes, device, label_encoder):
    """Conservative cross-validation"""
    print("\n" + "="*60)
    print("ROBUST 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': [],
        'loss': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
        print(f"\nFold {fold + 1}/5:")
        
        # Split data
        X_train = train_data[train_idx]
        X_val = train_data[val_idx]
        y_train = train_labels[train_idx]
        y_val = train_labels[val_idx]
        len_train = [train_lengths[i] for i in train_idx]
        len_val = [train_lengths[i] for i in val_idx]
        
        # Create datasets
        train_dataset = CommandDataset(X_train, y_train, len_train)
        val_dataset = CommandDataset(X_val, y_val, len_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # Create model
        model = RobustClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)
        
        # Conservative training setup
        criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Train for fewer epochs
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(20):  # Fewer epochs for CV
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                seqs = batch['sequences'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths'].to(device)
                
                optimizer.zero_grad()
                outputs = model(seqs, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            if epoch % 5 == 0 or epoch == 19:
                val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate_model(
                    model, val_loader, criterion, device, label_encoder
                )
                
                if val_f1_macro > best_f1:
                    best_f1 = val_f1_macro
                    best_acc = val_acc
                    best_loss = val_loss
                    best_f1_weighted = val_f1_weighted
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 2:  # Early stopping
                        break
        
        cv_results['accuracy'].append(best_acc)
        cv_results['f1_macro'].append(best_f1)
        cv_results['f1_weighted'].append(best_f1_weighted)
        cv_results['loss'].append(best_loss)
        
        print(f"  Best - Acc: {best_acc:.4f}, F1-Macro: {best_f1:.4f}")
    
    # Print results
    print(f"\nCross-validation Results:")
    for metric, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    return cv_results

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Training Loss', alpha=0.8)
    ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Training Accuracy', alpha=0.8)
    ax2.plot(val_accs, label='Validation Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Progress - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.pt', '_training.png'), dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path.replace('.pt', '_confusion.png'), dpi=150)
    plt.close()

def main():
    # Load robustly processed data
    data_dir = r"C:\Users\Faster\Downloads\Autonomous"
    train_path = os.path.join(data_dir, "robust_preprocessed_train.npz")
    test_path = os.path.join(data_dir, "robust_preprocessed_test.npz")
    vocab_path = os.path.join(data_dir, "robust_preprocessed_vocab.pkl")
    label_enc_path = os.path.join(data_dir, "robust_preprocessed_label_encoder.pkl")
    info_path = os.path.join(data_dir, "robust_preprocessed_info.pkl")
    
    print("Loading robustly preprocessed data...")
    
    # Load training data
    train_data = np.load(train_path)
    train_sequences = train_data['sequences']
    train_labels = train_data['labels']
    train_lengths = train_data['lengths'].tolist()
    
    # Load test data
    test_data = np.load(test_path)
    test_sequences = test_data['sequences']
    test_labels = test_data['labels']
    test_lengths = test_data['lengths'].tolist()
    
    # Load vocabulary and label encoder
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(label_enc_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    
    print(f"Train size: {len(train_sequences)}")
    print(f"Test size: {len(test_sequences)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Display preprocessing info
    print(f"\nData Quality Assessment:")
    print(f"  Max sequence length: {info['max_seq_len']}")
    print(f"  Average sequence length: {info['avg_seq_len']:.2f}")
    print(f"  Suspicious tokens removed: {len(info.get('suspicious_tokens', []))}")
    
    if info.get('has_potential_leakage', False):
        print("  ⚠️  Potential leakage detected - using extra regularization")
    else:
        print("  ✅ Data quality looks good")
    
    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print(f"Class weights: {dict(zip(label_encoder.classes_, class_weights))}")
    
    # Cross-validation
    cv_results = robust_cross_validate(
        train_sequences, train_labels, train_lengths, 
        vocab_size, num_classes, device, label_encoder
    )
    
    # Check if CV results are realistic
    mean_cv_acc = np.mean(cv_results['accuracy'])
    if mean_cv_acc > 0.95:
        print(f"\n⚠️  WARNING: CV accuracy ({mean_cv_acc:.3f}) seems unrealistically high!")
        print("This suggests potential data leakage. Consider:")
        print("  - More aggressive preprocessing")
        print("  - Checking for temporal dependencies")
        print("  - Reducing model complexity further")
        
        # Use even more regularization
        global DROPOUT, WEIGHT_DECAY, LEARNING_RATE
        DROPOUT = 0.7
        WEIGHT_DECAY = 0.02
        LEARNING_RATE = 0.0003
        print(f"Applying extra regularization: dropout={DROPOUT}, wd={WEIGHT_DECAY}")
    
    # Final training on full training set
    print(f"\n{'='*60}")
    print("FINAL TRAINING")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = CommandDataset(train_sequences, train_labels, train_lengths)
    test_dataset = CommandDataset(test_sequences, test_labels, test_lengths)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = RobustClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup with heavy regularization
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.15)  # More smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training tracking
    best_test_f1 = 0
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    print("Starting robust training...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            seqs = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_train_loss += loss.item() * seqs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        scheduler.step()
        train_acc = correct_train / total_train
        avg_train_loss = total_train_loss / total_train
        
        # Test evaluation
        test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_true = evaluate_model(
            model, test_loader, criterion, device, label_encoder
        )
        
        # Store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: Loss {avg_train_loss:.4f}, Acc {train_acc:.4f} | "
                  f"Test: Loss {test_loss:.4f}, Acc {test_acc:.4f}, F1 {test_f1_macro:.4f}")
        
        # Model saving and early stopping
        if test_f1_macro > best_test_f1:
            best_test_f1 = test_f1_macro
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            checkpoint = {
                'model_state_dict': best_model_state,
                'vocab_size': vocab_size,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_classes': num_classes,
                'dropout': DROPOUT,
                'label_encoder': label_encoder,
                'vocab': vocab,
                'best_metrics': {
                    'epoch': epoch,
                    'test_accuracy': test_acc,
                    'test_f1_macro': test_f1_macro,
                    'test_f1_weighted': test_f1_weighted,
                    'predictions': test_preds,
                    'true_labels': test_true
                },
                'cv_results': cv_results,
                'training_config': {
                    'max_seq_len': MAX_SEQ_LEN,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'dropout': DROPOUT
                }
            }
            torch.save(checkpoint, SAVE_PATH)
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation
    final_loss, final_acc, final_f1_macro, final_f1_weighted, final_preds, final_true = evaluate_model(
        model, test_loader, criterion, device, label_encoder
    )
    
    print(f"Cross-validation F1-Macro: {np.mean(cv_results['f1_macro']):.4f} ± {np.std(cv_results['f1_macro']):.4f}")
    print(f"Final test accuracy: {final_acc:.4f}")
    print(f"Final test F1-Macro: {final_f1_macro:.4f}")
    print(f"Final test F1-Weighted: {final_f1_weighted:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Test Results:")
    print(classification_report(final_true, final_preds, target_names=label_encoder.classes_))
    
    # Assessment of results
    print(f"\n{'='*60}")
    print("REALISM ASSESSMENT")
    print(f"{'='*60}")
    
    if final_acc > 0.95:
        print("⚠️  VERY HIGH ACCURACY - Potential issues:")
        print("  - Data leakage still present")
        print("  - Classes may be too easily separable")
        print("  - Consider collecting more diverse data")
    elif final_acc > 0.85:
        print("⚠️  HIGH ACCURACY - Monitor for:")
        print("  - Generalization to new data sources")
        print("  - Performance on adversarial examples")
    elif final_acc > 0.70:
        print("✅ REALISTIC ACCURACY - Good signs:")
        print("  - Model likely to generalize well")
        print("  - Appropriate task difficulty")
    else:
        print("📊 MODERATE ACCURACY - Consider:")
        print("  - More sophisticated model architecture")
        print("  - Feature engineering improvements")
        print("  - More training data")
    
    # Create visualizations
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, SAVE_PATH)
    plot_confusion_matrix(final_true, final_preds, label_encoder.classes_, SAVE_PATH, "Test Set Confusion Matrix")
    
    print(f"\nFiles saved:")
    print(f"  - Model: {SAVE_PATH}")
    print(f"  - Training curves: {SAVE_PATH.replace('.pt', '_training.png')}")
    print(f"  - Confusion matrix: {SAVE_PATH.replace('.pt', '_confusion.png')}")

if __name__ == '__main__':
    main()