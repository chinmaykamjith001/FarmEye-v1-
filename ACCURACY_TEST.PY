import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
import json
warnings.filterwarnings('ignore')

class CommandDataset(Dataset):
    """Dataset class for loading sequences"""
    def __init__(self, sequences, labels, lengths, max_len=25):
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
        
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
            length = self.max_len
        
        return {
            'sequence': torch.tensor(seq, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': length
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = [item['length'] for item in batch]
    
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
    """Model architecture (must match training script)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout)
        
        self.attention = nn.Linear(hidden_dim, 1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, lengths=None):
        batch_size, seq_len = x.size()
        
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.lstm_dropout(lstm_out)
        
        if lengths is not None:
            mask = torch.zeros(batch_size, seq_len, device=x.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1.0
            
            attn_weights = self.attention(lstm_out).squeeze(-1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            output = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        else:
            output = lstm_out.mean(dim=1)
        
        output = self.classifier(output)
        return output

def load_model_and_data_safe(model_path, data_dir):
    """Load trained model and test data with safe loading"""
    print("Loading model and data...")
    
    # SOLUTION 1: Use safe globals context manager (recommended)
    from sklearn.preprocessing import LabelEncoder
    with torch.serialization.safe_globals([LabelEncoder]):
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model parameters
    vocab_size = checkpoint['vocab_size']
    embed_dim = checkpoint['embed_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_classes = checkpoint['num_classes']
    dropout = checkpoint['dropout']
    
    # Load model
    model = RobustClassifier(vocab_size, embed_dim, hidden_dim, num_classes, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_path = os.path.join(data_dir, "robust_preprocessed_test.npz")
    test_data = np.load(test_path)
    test_sequences = test_data['sequences']
    test_labels = test_data['labels']
    test_lengths = test_data['lengths'].tolist()
    
    # Load label encoder and vocab
    label_encoder = checkpoint['label_encoder']
    vocab = checkpoint['vocab']
    
    print(f"Model loaded:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {label_encoder.classes_}")
    print(f"Test data: {len(test_sequences)} samples")
    
    return model, test_sequences, test_labels, test_lengths, label_encoder, vocab, checkpoint

def load_model_and_data_legacy(model_path, data_dir):
    """Load trained model and test data with legacy loading (less secure)"""
    print("Loading model and data...")
    
    # SOLUTION 2: Use weights_only=False (less secure but simpler)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model parameters
    vocab_size = checkpoint['vocab_size']
    embed_dim = checkpoint['embed_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_classes = checkpoint['num_classes']
    dropout = checkpoint['dropout']
    
    # Load model
    model = RobustClassifier(vocab_size, embed_dim, hidden_dim, num_classes, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_path = os.path.join(data_dir, "robust_preprocessed_test.npz")
    test_data = np.load(test_path)
    test_sequences = test_data['sequences']
    test_labels = test_data['labels']
    test_lengths = test_data['lengths'].tolist()
    
    # Load label encoder and vocab
    label_encoder = checkpoint['label_encoder']
    vocab = checkpoint['vocab']
    
    print(f"Model loaded:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {label_encoder.classes_}")
    print(f"Test data: {len(test_sequences)} samples")
    
    return model, test_sequences, test_labels, test_lengths, label_encoder, vocab, checkpoint

def comprehensive_evaluation(model, test_loader, label_encoder, device):
    """Perform comprehensive model evaluation"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            seqs = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            outputs = model(seqs, lengths)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            confidences = probs.max(dim=1)[0]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_confidences = np.array(all_confidences)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    
    print(f"Overall Metrics:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  F1-Score (Macro):   {f1_macro:.4f}")
    print(f"  F1-Score (Weighted):{f1_weighted:.4f}")
    print(f"  Precision (Macro):  {precision_macro:.4f}")
    print(f"  Recall (Macro):     {recall_macro:.4f}")
    print(f"  Average Confidence: {np.mean(all_confidences):.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'confidences': all_confidences,
        'per_class_f1': f1_per_class,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class
    }

def analyze_prediction_confidence(results, label_encoder, save_dir):
    """Analyze prediction confidence patterns"""
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS")
    print("="*60)
    
    confidences = results['confidences']
    predictions = results['predictions']
    true_labels = results['true_labels']
    
    # Overall confidence statistics
    print(f"Confidence Statistics:")
    print(f"  Mean:   {np.mean(confidences):.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    print(f"  Std:    {np.std(confidences):.4f}")
    print(f"  Min:    {np.min(confidences):.4f}")
    print(f"  Max:    {np.max(confidences):.4f}")
    
    # Confidence by correctness
    correct_mask = predictions == true_labels
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    print(f"\nConfidence by Correctness:")
    if len(correct_confidences) > 0:
        print(f"  Correct predictions:   {np.mean(correct_confidences):.4f} ± {np.std(correct_confidences):.4f}")
    if len(incorrect_confidences) > 0:
        print(f"  Incorrect predictions: {np.mean(incorrect_confidences):.4f} ± {np.std(incorrect_confidences):.4f}")
    
    # Confidence distribution plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    if len(correct_confidences) > 0:
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', density=True)
    if len(incorrect_confidences) > 0:
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution by Correctness')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    
    accuracies = []
    for i in range(len(confidence_bins) - 1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == true_labels[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    plt.plot(bin_centers, accuracies, 'o-', label='Actual Accuracy')
    plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Model Calibration')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=150)
    plt.close()
    
    return {
        'mean_confidence': np.mean(confidences),
        'correct_confidence': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
        'incorrect_confidence': np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0
    }

def create_detailed_confusion_matrix(results, label_encoder, save_dir):
    """Create detailed confusion matrix with percentages"""
    predictions = results['predictions']
    true_labels = results['true_labels']
    class_names = label_encoder.classes_
    
    cm = confusion_matrix(true_labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    # Create annotation text
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            row.append(f'{count}\n({percentage:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'label': 'Percentage'})
    
    plt.title('Detailed Confusion Matrix\n(Count and Percentage)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_confusion_matrix.png'), dpi=150)
    plt.close()

def error_analysis(results, test_sequences, test_lengths, vocab, label_encoder, save_dir):
    """Analyze prediction errors in detail"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    predictions = results['predictions']
    true_labels = results['true_labels']
    probabilities = results['probabilities']
    confidences = results['confidences']
    
    # Find incorrect predictions
    incorrect_mask = predictions != true_labels
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print(f"Total errors: {len(incorrect_indices)} out of {len(predictions)} ({len(incorrect_indices)/len(predictions)*100:.2f}%)")
    
    if len(incorrect_indices) == 0:
        print("No errors found!")
        return
    
    # Create reverse vocabulary
    idx2token = {idx: token for token, idx in vocab.items()}
    
    # Analyze worst predictions (lowest confidence incorrect predictions)
    incorrect_confidences = confidences[incorrect_mask]
    worst_indices = incorrect_indices[np.argsort(incorrect_confidences)[:min(10, len(incorrect_indices))]]
    
    print(f"\nWorst Predictions (lowest confidence errors):")
    print("-" * 60)
    
    error_details = []
    for i, idx in enumerate(worst_indices):
        seq = test_sequences[idx]
        length = test_lengths[idx]
        true_class = label_encoder.classes_[true_labels[idx]]
        pred_class = label_encoder.classes_[predictions[idx]]
        confidence = confidences[idx]
        prob_dist = probabilities[idx]
        
        # Convert sequence to tokens
        tokens = []
        for token_idx in seq[:length]:
            if token_idx in idx2token:
                tokens.append(idx2token[token_idx])
        
        token_str = ' '.join(tokens)
        
        print(f"\nExample {i+1}:")
        print(f"  True:       {true_class}")
        print(f"  Predicted:  {pred_class} (confidence: {confidence:.3f})")
        print(f"  Sequence:   {token_str}")
        print(f"  Prob dist:  {[f'{label_encoder.classes_[j]}: {prob_dist[j]:.3f}' for j in range(len(label_encoder.classes_))]}")
        
        error_details.append({
            'index': int(idx),
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': float(confidence),
            'tokens': token_str,
            'probability_distribution': {label_encoder.classes_[j]: float(prob_dist[j]) for j in range(len(label_encoder.classes_))}
        })
    
    # Save error analysis
    with open(os.path.join(save_dir, 'error_analysis.json'), 'w') as f:
        json.dump(error_details, f, indent=2)
    
    # Error patterns by class
    print(f"\nError Patterns by Class:")
    for true_class_idx, true_class in enumerate(label_encoder.classes_):
        class_mask = true_labels == true_class_idx
        class_errors = incorrect_mask & class_mask
        if np.sum(class_mask) > 0:
            error_rate = np.sum(class_errors) / np.sum(class_mask)
            print(f"  {true_class}: {np.sum(class_errors)}/{np.sum(class_mask)} errors ({error_rate:.2%})")
            
            # Most confused with
            if np.sum(class_errors) > 0:
                confused_preds = predictions[class_errors]
                most_confused = np.bincount(confused_preds).argmax()
                print(f"    Most confused with: {label_encoder.classes_[most_confused]}")

def generate_test_report(results, checkpoint, confidence_analysis, save_dir):
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    # Compile all results
    report = {
        'model_info': {
            'vocabulary_size': checkpoint['vocab_size'],
            'embedding_dimension': checkpoint['embed_dim'],
            'hidden_dimension': checkpoint['hidden_dim'],
            'number_of_classes': checkpoint['num_classes'],
            'dropout_rate': checkpoint['dropout'],
            'classes': checkpoint['label_encoder'].classes_.tolist()
        },
        'training_info': checkpoint.get('training_config', {}),
        'test_performance': {
            'accuracy': float(results['accuracy']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
            'precision_macro': float(results['precision_macro']),
            'recall_macro': float(results['recall_macro'])
        },
        'per_class_performance': {
            checkpoint['label_encoder'].classes_[i]: {
                'precision': float(results['per_class_precision'][i]),
                'recall': float(results['per_class_recall'][i]),
                'f1_score': float(results['per_class_f1'][i])
            }
            for i in range(len(checkpoint['label_encoder'].classes_))
        },
        'confidence_analysis': confidence_analysis,
        'cross_validation_results': checkpoint.get('cv_results', {}),
        'assessment': {}
    }
    
    # Performance assessment
    accuracy = results['accuracy']
    if accuracy > 0.95:
        report['assessment']['level'] = 'VERY_HIGH'
        report['assessment']['note'] = 'Accuracy may be unrealistically high - check for data leakage'
        report['assessment']['concerns'] = [
            'Potential data leakage',
            'Classes may be too easily separable',
            'May not generalize to real-world data'
        ]
    elif accuracy > 0.85:
        report['assessment']['level'] = 'HIGH'
        report['assessment']['note'] = 'Good performance - monitor generalization'
        report['assessment']['concerns'] = [
            'Monitor performance on new data sources',
            'Test on adversarial examples'
        ]
    elif accuracy > 0.70:
        report['assessment']['level'] = 'REALISTIC'
        report['assessment']['note'] = 'Realistic performance for this task'
        report['assessment']['strengths'] = [
            'Likely to generalize well',
            'Appropriate task difficulty'
        ]
    else:
        report['assessment']['level'] = 'MODERATE'
        report['assessment']['note'] = 'Room for improvement'
        report['assessment']['suggestions'] = [
            'Consider model architecture improvements',
            'More feature engineering',
            'Additional training data'
        ]
    
    # Save report
    report_path = os.path.join(save_dir, 'test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"Test Report Summary:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Assessment Level: {report['assessment']['level']}")
    print(f"  Note: {report['assessment']['note']}")
    
    return report

def main():
    """Main testing function"""
    # Configuration
    model_path = r"C:\Users\Faster\Downloads\Autonomous\balanced_enhanced_classifier.pt"
    data_dir = r"C:\Users\Faster\Downloads\Autonomous"
    save_dir = data_dir
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Make sure you've trained the model first using the robust trainer script.")
        return
    
    # Load model and data
    try:
        # Try the safe loading method first (recommended)
        print("Attempting secure loading with safe globals...")
        model, test_sequences, test_labels, test_lengths, label_encoder, vocab, checkpoint = load_model_and_data_safe(
            model_path, data_dir
        )
    except Exception as e:
        print(f"Safe loading failed: {e}")
        try:
            # Fall back to legacy loading if safe loading fails
            print("Falling back to legacy loading (weights_only=False)...")
            model, test_sequences, test_labels, test_lengths, label_encoder, vocab, checkpoint = load_model_and_data_legacy(
                model_path, data_dir
            )
        except Exception as e2:
            print(f"Error loading model or data: {e2}")
            return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Create test dataset and loader
    test_dataset = CommandDataset(test_sequences, test_labels, test_lengths)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Comprehensive evaluation
    results = comprehensive_evaluation(model, test_loader, label_encoder, device)
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(results['true_labels'], results['predictions'], 
                              target_names=label_encoder.classes_))
    
    # Confidence analysis
    confidence_analysis = analyze_prediction_confidence(results, label_encoder, save_dir)
    
    # Create detailed confusion matrix
    create_detailed_confusion_matrix(results, label_encoder, save_dir)
    
    # Error analysis
    error_analysis(results, test_sequences, test_lengths, vocab, label_encoder, save_dir)
    
    # Generate comprehensive report
    report = generate_test_report(results, checkpoint, confidence_analysis, save_dir)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Files generated:")
    print(f"  - Test report: {os.path.join(save_dir, 'test_report.json')}")
    print(f"  - Confidence analysis: {os.path.join(save_dir, 'confidence_analysis.png')}")
    print(f"  - Confusion matrix: {os.path.join(save_dir, 'detailed_confusion_matrix.png')}")
    print(f"  - Error analysis: {os.path.join(save_dir, 'error_analysis.json')}")

if __name__ == '__main__':
    main()