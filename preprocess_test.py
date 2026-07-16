#preprocess script

import os
import re
import pandas as pd
import pickle
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def aggressive_clean(text):
    """Aggressive cleaning to remove class-specific artifacts"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # 1. Remove ALL prompt indicators and user-specific info
    prompt_patterns = [
        r'^.*[@#$%>:]+.*$',                    # Any line with prompt chars
        r'\b[a-zA-Z0-9\-\_]+@[a-zA-Z0-9\-\.]+\b',  # email-like patterns
        r'\[[^\]]*\]',                         # [anything]
        r'\b(root|admin|user|guest|daemon|ubuntu|kali|parrot)\b',  # common usernames
        r'\$\s*$',                             # trailing prompts
        r'#\s*$',                              # trailing prompts
    ]
    
    for pattern in prompt_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 2. Aggressively remove timestamps, IDs, and session info
    time_patterns = [
        r'\b\d{1,2}:\d{2}:\d{2}\b',           # timestamps
        r'\b\d{4}-\d{2}-\d{2}\b',             # dates  
        r'\b\d{2}/\d{2}/\d{4}\b',             # dates
        r'\bpid\s*:?\s*\d+\b',                # process IDs
        r'\btty\d+\b',                        # TTY identifiers
        r'\bsession\s*:?\s*\d+\b',            # session IDs
        r'\b\d{10,}\b',                       # long numbers (likely timestamps)
    ]
    
    for pattern in time_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 3. Normalize ALL paths and network info
    text = re.sub(r'/\w+/[/\w\-\.\_]*', '/path/', text)  # Any absolute path
    text = re.sub(r'~/[/\w\-\.\_]*', '~/path/', text)    # Home paths
    text = re.sub(r'\./[/\w\-\.\_]*', './path/', text)   # Relative paths
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IPADDR', text)  # IP addresses
    text = re.sub(r':\d{1,5}\b', ':PORT', text)          # Port numbers
    text = re.sub(r'https?://[^\s]+', 'URL', text)       # URLs
    
    # 4. Remove tool-specific output and error messages
    output_patterns = [
        r'command not found',
        r'permission denied',
        r'no such file',
        r'access denied',
        r'connection refused',
        r'timeout',
        r'failed',
        r'success',
        r'complete',
        r'done',
        r'error',
        r'warning',
        r'info',
        r'\berror\s*:\s*',
        r'\bwarning\s*:\s*',
        r'^\s*\+.*$',                         # diff output
        r'^\s*\-.*$',                         # diff output
        r'^\s*@.*$',                          # diff headers
    ]
    
    for pattern in output_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 5. Remove obvious attack/recon indicators (major source of leakage)
    attack_terms = [
        r'\b(nmap|nikto|sqlmap|metasploit|hydra|john|hashcat|aircrack|wireshark)\b',
        r'\b(exploit|payload|shell|backdoor|trojan|malware|virus)\b',
        r'\b(scan|probe|enum|brute|crack|hack|attack|intrusion)\b',
        r'\b(vulnerable|vuln|cve-\d+|exploit-db)\b',
        r'\b(reverse|bind)[\s\-_]shell\b',
        r'\b(sql[\s\-_]injection|xss|csrf|lfi|rfi)\b',
        r'\b(pentesting|pentest|red[\s\-_]team|ctf)\b',
    ]
    
    for pattern in attack_terms:
        text = re.sub(pattern, 'TOOL', text, flags=re.IGNORECASE)
    
    # 6. Normalize common commands to reduce class-specific bias
    command_normalizations = {
        r'\b(ls|dir|ll)\b': 'list',
        r'\b(cat|type|more|less)\b': 'view',
        r'\b(cd|chdir|pushd|popd)\b': 'navigate',
        r'\b(cp|copy|mv|move|rm|del)\b': 'fileop',
        r'\b(grep|find|search|locate)\b': 'search',
        r'\b(wget|curl|download)\b': 'download',
        r'\b(ssh|telnet|nc|netcat)\b': 'connect',
        r'\b(ps|top|htop|tasklist)\b': 'process',
        r'\b(netstat|ss|lsof)\b': 'network',
        r'\b(sudo|su|runas)\b': 'elevate',
    }
    
    for pattern, replacement in command_normalizations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 7. Remove obvious noise and artifacts
    text = re.sub(r'\s*[;&|]+\s*', ' ', text)           # command separators
    text = re.sub(r'\s*2>/dev/null\s*', ' ', text)      # redirections
    text = re.sub(r'\s*>/dev/null\s*', ' ', text)       # redirections
    text = re.sub(r'\s*</dev/null\s*', ' ', text)       # redirections
    text = re.sub(r'\\x[0-9a-f]{2}', '', text)          # hex escapes
    text = re.sub(r'\\[rnt]', ' ', text)                # escape sequences
    text = re.sub(r'[^\w\s\-\./:]', ' ', text)          # keep only basic chars
    text = re.sub(r'\s+', ' ', text)                    # normalize whitespace
    
    return text.strip()

def extract_behavioral_features(text, max_length=20):
    """Extract behavioral patterns rather than specific commands"""
    if not text:
        return []
    
    tokens = text.split()
    if not tokens:
        return []
    
    # Focus on behavioral patterns
    behavioral_tokens = []
    
    for token in tokens[:max_length]:  # Limit sequence length
        # Skip obvious identifiers
        if len(token) < 2 or len(token) > 15:
            continue
        if token in ['IPADDR', 'PORT', 'URL', 'TOOL']:
            behavioral_tokens.append(token)
        elif token.startswith('/') or token.startswith('./') or token.startswith('~/'):
            continue  # Skip paths
        elif token.isdigit():
            behavioral_tokens.append('NUM')
        elif token.startswith('-') and len(token) <= 4:
            behavioral_tokens.append('FLAG')
        elif re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', token):
            behavioral_tokens.append(token)
    
    return behavioral_tokens

def build_conservative_vocab(token_lists, labels, min_freq=10, max_vocab=500):
    """Build vocabulary with strong anti-leakage measures"""
    
    # Count tokens by class
    class_tokens = defaultdict(Counter)
    overall_counter = Counter()
    
    for tokens, label in zip(token_lists, labels):
        class_tokens[label].update(tokens)
        overall_counter.update(tokens)
    
    print(f"Total unique tokens: {len(overall_counter)}")
    
    # Find highly discriminative tokens (likely leakage)
    suspicious_tokens = set()
    class_names = list(set(labels))
    
    for token, total_freq in overall_counter.items():
        if total_freq < min_freq:
            continue
            
        class_freqs = [class_tokens[cls][token] for cls in class_names]
        max_freq = max(class_freqs)
        other_freq = sum(class_freqs) - max_freq
        
        # Mark as suspicious if highly skewed towards one class
        if max_freq > other_freq * 2:  # Very conservative threshold
            suspicious_tokens.add(token)
    
    # Additional suspicious patterns
    suspicious_patterns = {
        'TOOL',  # Our normalized attack tools
        'attack', 'recon', 'benign',  # class names that might leak
        'high', 'low',  # severity indicators
    }
    suspicious_tokens.update(suspicious_patterns)
    
    # Build clean vocabulary
    vocab_tokens = ['<PAD>', '<UNK>']
    
    # Only keep tokens that appear in multiple classes with reasonable frequency
    for token, freq in overall_counter.most_common():
        if len(vocab_tokens) >= max_vocab:
            break
            
        if (token not in suspicious_tokens and 
            freq >= min_freq and 
            len(token) >= 2 and
            len(token) <= 12):
            
            # Check if token appears in multiple classes
            class_count = sum(1 for cls in class_names if class_tokens[cls][token] > 0)
            if class_count >= 2:  # Must appear in at least 2 classes
                vocab_tokens.append(token)
    
    token2idx = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    
    print(f"Final vocabulary size: {len(token2idx)}")
    print(f"Removed {len(suspicious_tokens)} suspicious tokens")
    print(f"Sample vocab: {vocab_tokens[2:12]}")
    
    return token2idx, suspicious_tokens

def analyze_class_distribution(df, token_col='tokens'):
    """Analyze class distribution and detect potential leakage"""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    class_tokens = defaultdict(Counter)
    total_by_class = defaultdict(int)
    
    for _, row in df.iterrows():
        label = row['label']
        tokens = row[token_col]
        class_tokens[label].update(tokens)
        total_by_class[label] += len(tokens)
    
    # Check class balance
    class_counts = df['label'].value_counts()
    print(f"Class distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Check for class-specific tokens
    suspicious_count = 0
    all_labels = list(class_tokens.keys())
    
    for label in all_labels:
        label_specific = []
        for token, count in class_tokens[label].most_common(20):
            other_count = sum(class_tokens[other][token] for other in all_labels if other != label)
            if other_count == 0 and count > 5:  # Only in this class
                label_specific.append(token)
                suspicious_count += 1
        
        if label_specific:
            print(f"\n⚠️  {label} has unique tokens: {label_specific[:10]}")
    
    return suspicious_count > 10  # Return True if high leakage risk

def robust_preprocess(csv_path, save_dir, test_size=0.2):
    """Robust preprocessing with anti-leakage measures"""
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, "robust_preprocessed")
    
    print("Loading and cleaning data...")
    df = pd.read_csv(csv_path)
    
    print(f"Original data shape: {df.shape}")
    df = df.dropna(subset=['tty_contents', 'label'])
    print(f"After removing NaN: {df.shape}")
    
    # Apply aggressive cleaning
    df['cleaned'] = df['tty_contents'].apply(aggressive_clean)
    df['tokens'] = df['cleaned'].apply(extract_behavioral_features)
    
    # Remove empty sequences
    df = df[df['tokens'].apply(lambda x: len(x) >= 3)]  # Minimum 3 tokens
    print(f"After removing short sequences: {df.shape}")
    
    # Check for class balance and potential leakage
    has_leakage = analyze_class_distribution(df, 'tokens')
    
    if has_leakage:
        print("\n⚠️  WARNING: Potential data leakage detected!")
        print("Consider additional data cleaning or collection")
    
    # Split data BEFORE building vocabulary to prevent leakage
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Build vocabulary ONLY on training data
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['label'])
    
    vocab, suspicious = build_conservative_vocab(
        train_df['tokens'].tolist(), 
        train_df['label'].tolist(),
        min_freq=15,  # Higher minimum frequency
        max_vocab=300  # Smaller vocabulary
    )
    
    # Convert sequences
    def tokens_to_sequence(tokens, vocab):
        return [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    train_sequences = [tokens_to_sequence(tokens, vocab) for tokens in train_df['tokens']]
    test_sequences = [tokens_to_sequence(tokens, vocab) for tokens in test_df['tokens']]
    test_labels = le.transform(test_df['label'])
    
    # Calculate statistics
    train_lengths = [len(seq) for seq in train_sequences]
    test_lengths = [len(seq) for seq in test_sequences]
    
    max_len = max(max(train_lengths), max(test_lengths))
    avg_len = np.mean(train_lengths + test_lengths)
    
    print(f"\nSequence statistics:")
    print(f"Max length: {max_len}")
    print(f"Average length: {avg_len:.2f}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Pad sequences
    def pad_sequences(sequences, max_len):
        padded = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            padded[i, :length] = seq[:length]
        return padded
    
    train_padded = pad_sequences(train_sequences, max_len)
    test_padded = pad_sequences(test_sequences, max_len)
    
    # Save data
    np.savez_compressed(
        save_prefix + '_train.npz',
        sequences=train_padded,
        labels=train_labels,
        lengths=train_lengths
    )
    
    np.savez_compressed(
        save_prefix + '_test.npz',
        sequences=test_padded,
        labels=test_labels,
        lengths=test_lengths
    )
    
    with open(save_prefix + '_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(save_prefix + '_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save preprocessing info
    info = {
        'vocab_size': len(vocab),
        'num_classes': len(le.classes_),
        'max_seq_len': max_len,
        'avg_seq_len': avg_len,
        'classes': le.classes_.tolist(),
        'train_size': len(train_sequences),
        'test_size': len(test_sequences),
        'suspicious_tokens': list(suspicious)[:50],
        'has_potential_leakage': has_leakage,
        'cleaning_examples': [
            {
                'original': train_df.iloc[i]['tty_contents'][:200],
                'cleaned': train_df.iloc[i]['cleaned'][:200],
                'tokens': train_df.iloc[i]['tokens']
            }
            for i in range(min(5, len(train_df)))
        ]
    }
    
    with open(save_prefix + '_info.pkl', 'wb') as f:
        pickle.dump(info, f)
    
    print(f"\n{'='*60}")
    print("ROBUST PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Max sequence length: {max_len}")
    print(f"Files saved:")
    print(f"  - {save_prefix}_train.npz")
    print(f"  - {save_prefix}_test.npz")
    print(f"  - {save_prefix}_vocab.pkl")
    print(f"  - {save_prefix}_label_encoder.pkl")
    print(f"  - {save_prefix}_info.pkl")
    
    if has_leakage:
        print(f"\n⚠️  Note: Monitor training carefully for overfitting")
    else:
        print(f"\n✅ Data appears suitable for realistic evaluation")

if __name__ == '__main__':
    csv_path = r"C:\Users\Faster\Downloads\Main_TTYs.csv"
    save_dir = r"C:\Users\Faster\Downloads\Autonomous"
    robust_preprocess(csv_path, save_dir)