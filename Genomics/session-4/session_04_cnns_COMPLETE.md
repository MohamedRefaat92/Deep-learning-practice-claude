# Session 4: Convolutional Neural Networks for Sequences
## PyTorch CNNs for Genomics - Motif Detection and Feature Extraction

**📖 Book References**: Deep Learning with PyTorch Ch. 8 (pages 245-290), Gen AI Ch. 4  
**⏱️ Duration**: 3-4 hours  
**🎯 Difficulty**: Intermediate

---

## 🎯 Learning Objectives & Core Functions

### What You'll Learn:
✅ 1D convolutions for DNA/protein sequences  
✅ Multi-kernel architectures for motif detection  
✅ Pooling operations (max, average, global)  
✅ Build DeepBind-style models  
✅ Visualize learned filters as PWMs  
✅ Extract sequence features automatically

### Core PyTorch Functions:
```python
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.MaxPool1d(kernel_size)
nn.AvgPool1d(kernel_size)
nn.AdaptiveMaxPool1d(output_size)
nn.BatchNorm1d(num_features)
F.conv1d()  # Functional interface
```

---

## 📚 Quick Theory

### Why CNNs for Sequences?

**DNA/Protein sequences have**:
- Local patterns (motifs, binding sites)
- Translation invariance (motifs can occur anywhere)
- Hierarchical features (motifs → domains → functions)

**CNNs provide**:
- Parameter sharing across positions
- Automatic feature learning
- Much fewer parameters than fully connected

### 1D Convolution:
```
Sequence: ATCGATCG (one-hot encoded to 4xL)
Kernel: 4x3 (4 channels, width 3)
Output: Detections at each position
```

---

## 🧪 Exercise 1: Basic 1D Convolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# One-hot encode DNA
def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding = np.zeros((4, len(seq)))
    for i, base in enumerate(seq):
        if base in mapping:
            encoding[mapping[base], i] = 1
    return torch.FloatTensor(encoding).unsqueeze(0)

# Test sequence
seq = "ATCGATCG" * 10
X = one_hot_encode(seq)
print(f"Encoded shape: {X.shape}")  # (1, 4, length)

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size=8):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 16, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        return x

model = SimpleCNN()
output = model(X)
print(f"Output shape: {output.shape}")
```

---

## 🧪 Exercise 2: Motif Detector

```python
# Generate data with known motif
def generate_sequences_with_motif(n_samples=1000, seq_len=50, motif="TATAAA"):
    sequences = []
    labels = []
    
    for _ in range(n_samples // 2):
        # Positive: contains motif
        pos = np.random.randint(0, seq_len - len(motif))
        seq = list(np.random.choice(['A','C','G','T'], seq_len))
        seq[pos:pos+len(motif)] = list(motif)
        sequences.append(''.join(seq))
        labels.append(1)
        
        # Negative: random
        seq = ''.join(np.random.choice(['A','C','G','T'], seq_len))
        sequences.append(seq)
        labels.append(0)
    
    return sequences, labels

sequences, labels = generate_sequences_with_motif()
X = torch.stack([one_hot_encode(s).squeeze(0) for s in sequences])
y = torch.FloatTensor(labels)

print(f"Data: {X.shape}, Labels: {y.shape}")

# CNN Motif Detector
class MotifCNN(nn.Module):
    def __init__(self, kernel_size=8, n_kernels=32):
        super().__init__()
        self.conv = nn.Conv1d(4, n_kernels, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_kernels, 1)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()

model = MotifCNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(50):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        acc = ((predictions > 0.5) == y).float().mean()
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
```

---

## 🧪 Exercise 3: Multi-Scale CNN

```python
class MultiScaleCNN(nn.Module):
    def __init__(self, kernel_sizes=[6, 8, 10, 12]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(4, 32, k) for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32 * len(kernel_sizes), 1)
        
    def forward(self, x):
        conv_outputs = []
        for conv in self.convs:
            out = F.relu(conv(x))
            out = self.pool(out)
            conv_outputs.append(out)
        
        x = torch.cat(conv_outputs, dim=1).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()

model = MultiScaleCNN()
print(model)

# Train similar to above
```

---

## 🧪 Exercise 4: Visualize Learned Filters

```python
def visualize_conv_filters(model):
    # Get first conv layer weights
    weights = model.conv.weight.data
    n_filters = weights.shape[0]
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Convert to PWM-like
            filter_weights = weights[i].numpy()
            ax.imshow(filter_weights, aspect='auto', cmap='RdBu_r')
            ax.set_title(f'Filter {i+1}')
            ax.set_yticks([0,1,2,3])
            ax.set_yticklabels(['A','C','G','T'])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

model = MotifCNN()
# Train model first...
visualize_conv_filters(model)
```

---

## 🧪 Exercise 5: Splice Site Prediction

```python
# More complex task: predict splice sites
class SpliceSiteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=11)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)  # 3 classes: donor, acceptor, neither
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SpliceSiteCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

---

## 🎯 Challenge: DeepBind Replication

Build a model similar to DeepBind for TF binding prediction.

```python
class DeepBindModel(nn.Module):
    def __init__(self, seq_len=101, n_motifs=16, motif_len=24):
        super().__init__()
        self.conv = nn.Conv1d(4, n_motifs, motif_len)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        
        # Calculate flattened size
        conv_out_len = seq_len - motif_len + 1
        pool_out_len = (conv_out_len - 3) // 3 + 1
        flat_size = n_motifs * pool_out_len
        
        self.fc1 = nn.Linear(flat_size, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

model = DeepBindModel()
# Train on ChIP-seq data...
```

---

## ✅ Self-Assessment

- [ ] Understand 1D convolution for sequences
- [ ] Can explain kernel size vs receptive field
- [ ] Know when to use pooling
- [ ] Built multi-kernel architectures
- [ ] Visualized learned filters
- [ ] Applied to real genomics tasks

---

## 📝 Key Takeaways

- **Conv1d**: Perfect for sequence data
- **Multiple kernels**: Detect different patterns
- **Pooling**: Downsample and aggregate
- **Adaptive pooling**: Fixed output size
- **Batch norm**: Faster, more stable training

---

*Session 4 Complete!*
