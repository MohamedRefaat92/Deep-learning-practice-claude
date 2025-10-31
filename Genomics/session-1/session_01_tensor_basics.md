# Practice Session 1: Tensor Basics and Operations
## PyTorch for Genomics - Foundational Skills

**Source**: Deep Learning with PyTorch (Second Edition), Chapter 3 - "It starts with a tensor"

**Duration**: 2-3 hours  
**Difficulty**: Beginner  
**Prerequisites**: Basic Python, understanding of DNA/RNA basics

---

## 🎯 Learning Objectives

By the end of this session, you will:
1. Create and manipulate PyTorch tensors representing genomic data
2. Understand tensor shapes, indexing, and slicing for sequence data
3. Perform vectorized operations on genomic datasets
4. Implement one-hot encoding for biological sequences
5. Work with batched genomic data efficiently

---

## 📚 Theory Review

### What are Tensors?
Tensors are multi-dimensional arrays that can run on GPUs. In genomics:
- **0D tensor (scalar)**: A single value (e.g., a gene expression measurement)
- **1D tensor (vector)**: A sequence (e.g., expression levels across samples)
- **2D tensor (matrix)**: Multiple sequences or expression matrix (genes × samples)
- **3D tensor (batch)**: Multiple matrices (batch × genes × samples)

### DNA Sequence Representation
DNA sequences can be encoded as:
1. **Integer encoding**: A=0, C=1, G=2, T=3
2. **One-hot encoding**: Each nucleotide becomes a 4-element vector
   - A = [1, 0, 0, 0]
   - C = [0, 1, 0, 0]
   - G = [0, 0, 1, 0]
   - T = [0, 0, 0, 1]

---

## 🧪 Exercise 1: DNA Sequence Encoding

### Part A: Basic Tensor Creation

```python
import torch
import numpy as np

# Create a DNA sequence
sequence = "ATCGATCGTTAGC"

# Task 1.1: Create a mapping dictionary
nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Task 1.2: Convert sequence to integer tensor
# YOUR CODE HERE
encoded_seq = torch.tensor([nucleotide_to_int[n] for n in sequence])

print(f"Original sequence: {sequence}")
print(f"Encoded tensor: {encoded_seq}")
print(f"Tensor shape: {encoded_seq.shape}")
print(f"Tensor dtype: {encoded_seq.dtype}")
```

**Expected Output:**
```
Original sequence: ATCGATCGTTAGC
Encoded tensor: tensor([0, 3, 1, 2, 0, 3, 1, 2, 3, 3, 0, 2, 1])
Tensor shape: torch.Size([13])
Tensor dtype: torch.int64
```

### Part B: One-Hot Encoding

```python
def one_hot_encode(sequence, nucleotide_to_int=None):
    """
    Convert a DNA sequence to one-hot encoded tensor.
    
    Args:
        sequence (str): DNA sequence string
        nucleotide_to_int (dict): Mapping of nucleotides to integers
        
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (4, seq_length)
    """
    if nucleotide_to_int is None:
        nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Task 1.3: Implement one-hot encoding
    # Hint: Use torch.zeros() and indexing
    # YOUR CODE HERE
    seq_length = len(sequence)
    one_hot = torch.zeros(4, seq_length)
    
    for i, nucleotide in enumerate(sequence):
        idx = nucleotide_to_int[nucleotide]
        one_hot[idx, i] = 1
    
    return one_hot

# Test your function
seq = "ATCG"
encoded = one_hot_encode(seq)
print(f"Sequence: {seq}")
print(f"One-hot encoded:\n{encoded}")
print(f"Shape: {encoded.shape}")
```

**Expected Output:**
```
Sequence: ATCG
One-hot encoded:
tensor([[1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [0., 1., 0., 0.]])
Shape: torch.Size([4, 4])
```

### Part C: Vectorized One-Hot Encoding (Advanced)

```python
def one_hot_encode_vectorized(sequence):
    """
    Faster vectorized one-hot encoding using PyTorch operations.
    
    Args:
        sequence (str): DNA sequence string
        
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (4, seq_length)
    """
    # Task 1.4: Implement using torch.nn.functional.one_hot
    # YOUR CODE HERE
    nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Convert to integer tensor
    int_seq = torch.tensor([nucleotide_to_int[n] for n in sequence])
    
    # Use one_hot function
    one_hot = torch.nn.functional.one_hot(int_seq, num_classes=4)
    
    # Transpose to get (4, seq_length) shape
    return one_hot.T.float()

# Test
seq = "ATCG"
encoded = one_hot_encode_vectorized(seq)
print(f"Vectorized one-hot:\n{encoded}")

# Benchmark (optional)
import time
long_seq = "ATCG" * 1000

start = time.time()
_ = one_hot_encode(long_seq)
time_loop = time.time() - start

start = time.time()
_ = one_hot_encode_vectorized(long_seq)
time_vectorized = time.time() - start

print(f"\nLoop version: {time_loop:.4f}s")
print(f"Vectorized version: {time_vectorized:.4f}s")
print(f"Speedup: {time_loop/time_vectorized:.2f}x")
```

---

## 🧪 Exercise 2: Gene Expression Matrix Operations

Gene expression data is typically stored as a matrix where:
- Rows = genes (e.g., 20,000 genes)
- Columns = samples (e.g., 100 samples)

### Part A: Creating and Exploring Expression Data

```python
import torch

# Simulate gene expression data
torch.manual_seed(42)  # For reproducibility

num_genes = 1000
num_samples = 50

# Task 2.1: Create random expression matrix
# Use torch.randn for normally distributed data
# YOUR CODE HERE
expression_matrix = torch.randn(num_genes, num_samples)

# Add some structure: make first 100 genes higher in first 25 samples
expression_matrix[:100, :25] += 2.0

print(f"Expression matrix shape: {expression_matrix.shape}")
print(f"Data type: {expression_matrix.dtype}")
print(f"Mean expression: {expression_matrix.mean():.3f}")
print(f"Std expression: {expression_matrix.std():.3f}")
print(f"\nFirst 5 genes, first 5 samples:")
print(expression_matrix[:5, :5])
```

### Part B: Normalization

```python
def normalize_expression(expr_matrix, method='zscore'):
    """
    Normalize gene expression matrix.
    
    Args:
        expr_matrix (torch.Tensor): Expression matrix (genes × samples)
        method (str): 'zscore' or 'minmax'
        
    Returns:
        torch.Tensor: Normalized expression matrix
    """
    # Task 2.2: Implement z-score normalization (per gene)
    # Formula: (x - mean) / std
    # YOUR CODE HERE
    
    if method == 'zscore':
        # Normalize each gene (row) independently
        mean = expr_matrix.mean(dim=1, keepdim=True)
        std = expr_matrix.std(dim=1, keepdim=True)
        normalized = (expr_matrix - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
        
    elif method == 'minmax':
        # Task 2.3: Implement min-max normalization
        # Formula: (x - min) / (max - min)
        # YOUR CODE HERE
        min_val = expr_matrix.min(dim=1, keepdim=True)[0]
        max_val = expr_matrix.max(dim=1, keepdim=True)[0]
        normalized = (expr_matrix - min_val) / (max_val - min_val + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

# Test normalization
normalized_zscore = normalize_expression(expression_matrix, method='zscore')
normalized_minmax = normalize_expression(expression_matrix, method='minmax')

print("Z-score normalized:")
print(f"  Mean: {normalized_zscore.mean():.6f}")
print(f"  Std: {normalized_zscore.std():.6f}")

print("\nMin-max normalized:")
print(f"  Min: {normalized_minmax.min():.6f}")
print(f"  Max: {normalized_minmax.max():.6f}")
```

### Part C: Statistical Analysis

```python
# Task 2.4: Calculate various statistics
# YOUR CODE HERE

# Per-gene statistics
gene_means = expression_matrix.mean(dim=1)  # Mean across samples
gene_stds = expression_matrix.std(dim=1)    # Std across samples

# Per-sample statistics  
sample_means = expression_matrix.mean(dim=0)  # Mean across genes
sample_stds = expression_matrix.std(dim=0)    # Std across genes

print("Per-gene statistics:")
print(f"  Mean of gene means: {gene_means.mean():.3f}")
print(f"  Mean of gene stds: {gene_stds.mean():.3f}")

print("\nPer-sample statistics:")
print(f"  Mean of sample means: {sample_means.mean():.3f}")
print(f"  Mean of sample stds: {sample_stds.mean():.3f}")

# Task 2.5: Find highly variable genes
# Genes with high std are more informative
top_k = 10
most_variable_genes = torch.argsort(gene_stds, descending=True)[:top_k]
print(f"\nTop {top_k} most variable genes (indices):")
print(most_variable_genes)
print(f"Their standard deviations:")
print(gene_stds[most_variable_genes])
```

---

## 🧪 Exercise 3: Sequence Manipulation and K-mers

### Part A: Reverse Complement

```python
def reverse_complement(sequence):
    """
    Compute reverse complement of DNA sequence.
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        str: Reverse complement
    """
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    # Task 3.1: Implement reverse complement
    # YOUR CODE HERE
    complement = ''.join([complement_map[n] for n in sequence])
    reverse_comp = complement[::-1]
    
    return reverse_comp

# Test
seq = "ATCGATCG"
rev_comp = reverse_complement(seq)
print(f"Original:  {seq}")
print(f"Rev Comp:  {rev_comp}")

# Verify with tensor operations
encoded = one_hot_encode_vectorized(seq)
rev_comp_encoded = one_hot_encode_vectorized(rev_comp)

print(f"\nOriginal encoded shape: {encoded.shape}")
print(f"Original:\n{encoded}")
print(f"\nReverse complement:\n{rev_comp_encoded}")
```

### Part B: K-mer Extraction

```python
def extract_kmers(sequence, k=6):
    """
    Extract all k-mers from a sequence.
    
    Args:
        sequence (str): DNA sequence
        k (int): K-mer length
        
    Returns:
        list: List of k-mers
    """
    # Task 3.2: Extract k-mers using sliding window
    # YOUR CODE HERE
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    
    return kmers

# Test
seq = "ATCGATCGTTA"
kmers = extract_kmers(seq, k=6)
print(f"Sequence: {seq}")
print(f"6-mers: {kmers}")
print(f"Number of 6-mers: {len(kmers)}")
```

### Part C: Batch K-mer Encoding

```python
def encode_kmer_batch(kmers):
    """
    Encode a batch of k-mers into a tensor.
    
    Args:
        kmers (list): List of k-mer strings
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 4, k)
    """
    # Task 3.3: Encode all k-mers and stack into batch
    # YOUR CODE HERE
    encoded_kmers = [one_hot_encode_vectorized(kmer) for kmer in kmers]
    batch = torch.stack(encoded_kmers, dim=0)
    
    return batch

# Test
seq = "ATCGATCGTTA"
kmers = extract_kmers(seq, k=6)
batch_tensor = encode_kmer_batch(kmers)

print(f"Batch tensor shape: {batch_tensor.shape}")
print(f"Shape interpretation: (num_kmers={batch_tensor.shape[0]}, "
      f"num_nucleotides={batch_tensor.shape[1]}, "
      f"kmer_length={batch_tensor.shape[2]})")

# Verify first k-mer
print(f"\nFirst k-mer: {kmers[0]}")
print(f"Encoded:\n{batch_tensor[0]}")
```

### Part D: Sliding Window with Stride

```python
def sliding_window_kmers(sequence, k=6, stride=1):
    """
    Extract k-mers with configurable stride.
    
    Args:
        sequence (str): DNA sequence
        k (int): K-mer length
        stride (int): Step size between k-mers
        
    Returns:
        torch.Tensor: Tensor of shape (num_windows, 4, k)
    """
    # Task 3.4: Implement sliding window with stride
    # YOUR CODE HERE
    kmers = []
    for i in range(0, len(sequence) - k + 1, stride):
        kmers.append(sequence[i:i+k])
    
    return encode_kmer_batch(kmers)

# Test with different strides
seq = "ATCGATCGATCGATCG"
k = 6

for stride in [1, 2, 3]:
    windows = sliding_window_kmers(seq, k=k, stride=stride)
    print(f"Stride {stride}: {windows.shape[0]} windows")
```

---

## 🧪 Exercise 4: Position Weight Matrix (PWM)

Position Weight Matrices are used to represent sequence motifs.

### Part A: Create PWM from Sequences

```python
def create_pwm(sequences):
    """
    Create a Position Weight Matrix from aligned sequences.
    
    Args:
        sequences (list): List of aligned DNA sequences (same length)
        
    Returns:
        torch.Tensor: PWM of shape (4, seq_length)
    """
    # Task 4.1: Count nucleotides at each position
    # YOUR CODE HERE
    
    if not sequences:
        raise ValueError("Empty sequence list")
    
    seq_length = len(sequences[0])
    num_seqs = len(sequences)
    
    # Initialize count matrix
    counts = torch.zeros(4, seq_length)
    nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Count occurrences
    for seq in sequences:
        for pos, nucleotide in enumerate(seq):
            idx = nucleotide_to_int[nucleotide]
            counts[idx, pos] += 1
    
    # Convert to frequencies (probabilities)
    pwm = counts / num_seqs
    
    return pwm

# Test with aligned sequences (binding sites)
binding_sites = [
    "ATCGAT",
    "ATCGAT",
    "ATCGAT",
    "ATGGAT",
    "ATCCAT",
    "ATCGAA",
]

pwm = create_pwm(binding_sites)
print("Position Weight Matrix:")
print(pwm)
print(f"\nShape: {pwm.shape}")

# Verify: each column should sum to 1
print(f"\nColumn sums (should be all 1.0):")
print(pwm.sum(dim=0))
```

### Part B: Score Sequence with PWM

```python
def score_with_pwm(sequence, pwm):
    """
    Score a sequence using a PWM.
    
    Args:
        sequence (str): DNA sequence
        pwm (torch.Tensor): PWM of shape (4, motif_length)
        
    Returns:
        float: PWM score
    """
    # Task 4.2: Calculate log-likelihood score
    # YOUR CODE HERE
    
    motif_length = pwm.shape[1]
    if len(sequence) != motif_length:
        raise ValueError(f"Sequence length must match PWM length ({motif_length})")
    
    nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    score = 0.0
    for pos, nucleotide in enumerate(sequence):
        idx = nucleotide_to_int[nucleotide]
        # Log-likelihood score (add pseudocount to avoid log(0))
        prob = pwm[idx, pos] + 1e-8
        score += torch.log2(prob).item()
    
    return score

# Test scoring
test_sequences = [
    "ATCGAT",  # Perfect match
    "ATGGAT",  # One mismatch
    "GGGGGG",  # Complete mismatch
]

print("Sequence scores:")
for seq in test_sequences:
    score = score_with_pwm(seq, pwm)
    print(f"  {seq}: {score:.3f}")
```

### Part C: Scan Sequence for Motif

```python
def scan_sequence(sequence, pwm, threshold=-10.0):
    """
    Scan a long sequence for motif occurrences.
    
    Args:
        sequence (str): Long DNA sequence to scan
        pwm (torch.Tensor): PWM of shape (4, motif_length)
        threshold (float): Minimum score for a hit
        
    Returns:
        list: List of (position, score) tuples for hits
    """
    # Task 4.3: Scan sequence and find high-scoring positions
    # YOUR CODE HERE
    
    motif_length = pwm.shape[1]
    hits = []
    
    for i in range(len(sequence) - motif_length + 1):
        subseq = sequence[i:i+motif_length]
        score = score_with_pwm(subseq, pwm)
        
        if score >= threshold:
            hits.append((i, score))
    
    return hits

# Test scanning
long_sequence = "GGGG" + "ATCGAT" + "AAAA" + "ATGGAT" + "TTTT" + "ATCGAT" + "CCCC"
hits = scan_sequence(long_sequence, pwm, threshold=-5.0)

print(f"Scanning sequence of length {len(long_sequence)}")
print(f"Found {len(hits)} hits:")
for pos, score in hits:
    subseq = long_sequence[pos:pos+6]
    print(f"  Position {pos}: {subseq} (score: {score:.3f})")
```

---

## 🧪 Exercise 5: Broadcasting and Batch Operations

Understanding broadcasting is crucial for efficient genomics computations.

### Part A: Broadcasting Basics

```python
import torch

# Task 5.1: Understand broadcasting with expression data
# YOUR CODE HERE

# Create expression matrix
expression = torch.randn(100, 20)  # 100 genes, 20 samples

# Calculate per-gene mean
gene_means = expression.mean(dim=1, keepdim=True)  # Shape: (100, 1)

# Center data (subtract mean from each sample)
centered = expression - gene_means  # Broadcasting!

print(f"Expression shape: {expression.shape}")
print(f"Gene means shape: {gene_means.shape}")
print(f"Centered shape: {centered.shape}")
print(f"\nVerify centering (means should be ~0):")
print(f"Centered gene means: {centered.mean(dim=1)[:5]}")
```

### Part B: Batch Distance Calculations

```python
def calculate_pairwise_distances(sequences_tensor):
    """
    Calculate pairwise Hamming distances between sequences.
    
    Args:
        sequences_tensor (torch.Tensor): Shape (num_sequences, 4, seq_length)
        
    Returns:
        torch.Tensor: Distance matrix of shape (num_sequences, num_sequences)
    """
    # Task 5.2: Implement using broadcasting
    # YOUR CODE HERE
    
    num_seqs = sequences_tensor.shape[0]
    
    # Expand dimensions for broadcasting
    # Shape: (num_seqs, 1, 4, seq_length)
    seq_a = sequences_tensor.unsqueeze(1)
    
    # Shape: (1, num_seqs, 4, seq_length)
    seq_b = sequences_tensor.unsqueeze(0)
    
    # Calculate element-wise differences and sum
    # Hamming distance = number of positions where sequences differ
    differences = (seq_a != seq_b).float()
    distances = differences.sum(dim=(2, 3))
    
    return distances

# Test with k-mers
kmers = ["ATCGAT", "ATCGAA", "GGGGGG", "ATCGAT"]
batch = encode_kmer_batch(kmers)

distances = calculate_pairwise_distances(batch)
print("Pairwise distances:")
print(distances)
print(f"\nExpected: identical sequences have distance 0")
print(f"Sequences 0 and 3 are identical: distance = {distances[0, 3]}")
```

---

## 🎯 Challenge Problems

### Challenge 1: Efficient Codon Translation

```python
def translate_to_protein(dna_sequence):
    """
    Translate DNA sequence to protein sequence.
    
    Args:
        dna_sequence (str): DNA sequence (length must be multiple of 3)
        
    Returns:
        str: Protein sequence (single letter amino acid codes)
    """
    # Task: Implement codon translation
    # Hint: Create a codon table dictionary
    # YOUR CODE HERE
    
    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    
    if len(dna_sequence) % 3 != 0:
        raise ValueError("DNA sequence length must be multiple of 3")
    
    protein = []
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3]
        amino_acid = codon_table.get(codon, 'X')  # X for unknown
        protein.append(amino_acid)
        if amino_acid == '*':  # Stop codon
            break
    
    return ''.join(protein)

# Test
dna = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
protein = translate_to_protein(dna)
print(f"DNA:     {dna}")
print(f"Protein: {protein}")
```

### Challenge 2: GC Content Analysis

```python
def analyze_gc_content(sequence, window_size=100, step=50):
    """
    Calculate GC content in sliding windows.
    
    Args:
        sequence (str): DNA sequence
        window_size (int): Window size
        step (int): Step size
        
    Returns:
        torch.Tensor: GC content values for each window
    """
    # Task: Calculate GC percentage in windows
    # YOUR CODE HERE
    
    gc_values = []
    
    for i in range(0, len(sequence) - window_size + 1, step):
        window = sequence[i:i+window_size]
        gc_count = window.count('G') + window.count('C')
        gc_percent = 100.0 * gc_count / window_size
        gc_values.append(gc_percent)
    
    return torch.tensor(gc_values)

# Test
long_seq = "AT" * 50 + "GC" * 50 + "AT" * 50  # Variable GC content
gc_content = analyze_gc_content(long_seq, window_size=50, step=25)

print(f"Sequence length: {len(long_seq)}")
print(f"Number of windows: {len(gc_content)}")
print(f"GC content per window:")
print(gc_content)
```

---

## ✅ Self-Assessment

Before moving to Session 2, ensure you can:

- [ ] Create tensors from genomic sequences
- [ ] Implement one-hot encoding efficiently
- [ ] Perform normalization on expression matrices
- [ ] Calculate statistics using tensor operations
- [ ] Understand and use tensor broadcasting
- [ ] Work with batched sequence data
- [ ] Implement PWM creation and scoring
- [ ] Extract and encode k-mers from sequences

---

## 📝 Additional Practice Ideas

1. **Load real data**: Download a FASTA file and process it with PyTorch
2. **Visualize PWMs**: Use matplotlib to create sequence logos
3. **Benchmark operations**: Compare NumPy vs PyTorch for genomic operations
4. **Add ambiguous bases**: Extend encoding to handle N (any base)
5. **RNA analysis**: Modify functions to work with RNA sequences (U instead of T)

---

## 🚀 Next Steps

Once you're comfortable with these exercises, move on to:
- **Session 2**: Autograd and gradient descent for genomics models
- Try implementing these functions with GPU acceleration using `.to('cuda')`
- Explore PyTorch's `torch.utils.data.Dataset` for loading large genomic files

---

## 📚 Additional Resources

- PyTorch Tensor Documentation: https://pytorch.org/docs/stable/tensors.html
- NumPy to PyTorch: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
- Biopython for sequence handling: https://biopython.org/

Good luck with your practice!
