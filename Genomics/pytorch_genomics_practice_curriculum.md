# PyTorch Deep Learning Practice Sessions for Genomics
## Curriculum for Bioinformaticians

---

## 📚 Learning Resources
- **Book 1**: Deep Learning with PyTorch (Second Edition) - Manning MEAP
- **Book 2**: Learn Generative AI with PyTorch - Mark Liu

---

## 🎯 Practice Session Overview

This curriculum progresses from foundational PyTorch skills to advanced genomics applications, with each session building upon previous concepts.

---

## **LEVEL 1: PyTorch Fundamentals**

### **Session 1: Tensor Basics and Operations**
**Source**: Deep Learning with PyTorch, Chapter 3 - "It starts with a tensor"

**Genomics Context**: Representing DNA sequences and gene expression data

**Learning Objectives:**
- Create and manipulate PyTorch tensors
- Understand tensor shapes, indexing, and slicing
- Perform basic operations on genomic data representations

**Practice Exercises:**

1. **DNA Sequence Encoding**
   - Create a one-hot encoding function for DNA sequences (A, T, G, C)
   - Convert a batch of DNA k-mers (length 6) into tensor format
   - Practice tensor reshaping and concatenation

2. **Gene Expression Matrix**
   - Load a simulated gene expression matrix (genes × samples)
   - Perform normalization using tensor operations
   - Calculate summary statistics (mean, std) across dimensions

3. **Sequence Manipulation**
   - Implement reverse complement using tensor operations
   - Create sliding windows over sequence tensors
   - Practice broadcasting with position weight matrices

**Code Starter:**
```python
import torch

# Exercise 1: DNA encoding
def one_hot_encode_sequence(sequence):
    # TODO: Implement one-hot encoding for DNA
    pass

# Exercise 2: Expression data
expression_data = torch.randn(1000, 50)  # 1000 genes, 50 samples
# TODO: Normalize and analyze

# Exercise 3: Sequence operations
dna_seq = "ATCGATCG"
# TODO: Implement reverse complement
```

---

### **Session 2: Autograd and Gradient Descent**
**Source**: Deep Learning with PyTorch, Chapter 5 - "The mechanics of learning"

**Genomics Context**: Optimizing predictive models for genomic features

**Learning Objectives:**
- Understand automatic differentiation
- Implement simple gradient descent
- Apply to basic genomics prediction tasks

**Practice Exercises:**

1. **Linear Model for Expression Prediction**
   - Build a linear model to predict gene expression from regulatory features
   - Manually implement gradient descent
   - Compare with PyTorch autograd

2. **Loss Function Exploration**
   - Implement MSE loss for regression tasks
   - Implement cross-entropy for variant classification
   - Visualize loss landscapes

3. **Learning Rate Optimization**
   - Experiment with different learning rates
   - Implement learning rate scheduling
   - Apply to a simple genomics dataset

**Code Starter:**
```python
import torch

# Exercise 1: Manual gradient descent
def predict_expression(features, weights, bias):
    return features @ weights + bias

# TODO: Implement training loop with manual gradients

# Exercise 2: With autograd
weights = torch.randn(10, 1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
# TODO: Implement using autograd
```

---

### **Session 3: Building Neural Networks**
**Source**: Deep Learning with PyTorch, Chapter 6 - "Using a neural network to fit the data"

**Genomics Context**: Creating models for sequence classification

**Learning Objectives:**
- Use nn.Module for model building
- Implement forward pass
- Understand layer types and activations

**Practice Exercises:**

1. **Promoter Classifier**
   - Build a 3-layer MLP to classify promoter vs non-promoter sequences
   - Use appropriate activation functions
   - Implement the model using nn.Module

2. **Gene Expression Regressor**
   - Create a neural network to predict gene expression from histone marks
   - Experiment with different architectures
   - Add dropout for regularization

3. **Multi-task Model**
   - Build a model with shared layers and multiple output heads
   - Predict both expression level and tissue type
   - Implement custom loss weighting

**Code Starter:**
```python
import torch
import torch.nn as nn

class PromoterClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # TODO: Define layers
        
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Exercise: Instantiate and test
model = PromoterClassifier(input_size=4*200, hidden_size=128)
```

---

## **LEVEL 2: Deep Learning Architectures**

### **Session 4: Convolutional Neural Networks**
**Source**: Deep Learning with PyTorch, Chapter 8 - "Using convolutions to generalize"

**Genomics Context**: Motif detection and sequence analysis

**Learning Objectives:**
- Understand 1D convolutions for sequences
- Implement CNNs for genomic data
- Use pooling and multiple convolutional layers

**Practice Exercises:**

1. **Motif Detector**
   - Build a 1D CNN to detect transcription factor binding motifs
   - Visualize learned filters
   - Interpret what the CNN has learned

2. **Splice Site Predictor**
   - Create a CNN to predict splice sites from sequence context
   - Use multiple convolutional layers with different kernel sizes
   - Implement global pooling

3. **DeepBind-style Model**
   - Replicate the architecture from DeepBind paper
   - Train on simulated protein-DNA binding data
   - Extract position weight matrices from filters

**Code Starter:**
```python
import torch.nn as nn

class MotifCNN(nn.Module):
    def __init__(self, seq_length, num_filters=32, kernel_size=12):
        super().__init__()
        # TODO: Implement 1D CNN architecture
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size)
        # Add more layers
        
    def forward(self, x):
        # x shape: (batch, 4, seq_length)
        # TODO: Implement forward pass
        pass
```

---

### **Session 5: Recurrent Neural Networks**
**Source**: Deep Learning with PyTorch, Chapter 9 - "Using PyTorch to fight cancer"

**Genomics Context**: Sequential data processing for genomics

**Learning Objectives:**
- Implement RNNs, LSTMs, and GRUs
- Handle variable-length sequences
- Apply to sequential genomic data

**Practice Exercises:**

1. **Variable-Length Sequence Classifier**
   - Build an LSTM to classify sequences of varying lengths
   - Implement sequence padding and packing
   - Handle batch processing efficiently

2. **Protein Sequence Property Predictor**
   - Use bidirectional LSTM for protein secondary structure prediction
   - Implement many-to-many architecture
   - Handle amino acid sequences

3. **Regulatory Element Scanner**
   - Create a GRU-based model to scan for regulatory elements
   - Implement attention mechanism
   - Visualize attention weights on sequences

**Code Starter:**
```python
import torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # TODO: Add output layers
        
    def forward(self, x, lengths):
        # TODO: Implement with packing/unpacking
        pass
```

---

### **Session 6: Transfer Learning**
**Source**: Deep Learning with PyTorch, Chapter 10 - "Combining data sources"

**Genomics Context**: Using pre-trained models and fine-tuning

**Learning Objectives:**
- Understand transfer learning concepts
- Fine-tune pre-trained models
- Adapt models to new genomic tasks

**Practice Exercises:**

1. **Fine-tune a Pre-trained CNN**
   - Start with a model trained on general sequence classification
   - Fine-tune for specific organism or tissue type
   - Implement freezing and unfreezing layers

2. **Domain Adaptation**
   - Train on mouse genomic data
   - Adapt to human genomic data with limited samples
   - Use appropriate loss functions

3. **Multi-organism Model**
   - Create a model that handles multiple species
   - Implement species-specific branches
   - Share common sequence processing layers

**Code Starter:**
```python
# Load pre-trained model
pretrained_model = torch.load('sequence_model.pt')

# Freeze early layers
for param in pretrained_model.conv_layers.parameters():
    param.requires_grad = False

# TODO: Add new classification head
# TODO: Implement fine-tuning strategy
```

---

## **LEVEL 3: Advanced Topics**

### **Session 7: Attention Mechanisms**
**Source**: Deep Learning with PyTorch, Advanced chapters

**Genomics Context**: Interpretable sequence models

**Learning Objectives:**
- Implement self-attention mechanisms
- Build transformer-style models
- Apply to genomic sequences

**Practice Exercises:**

1. **Sequence Attention Model**
   - Implement multi-head attention for DNA sequences
   - Visualize attention patterns
   - Identify important positions

2. **Cross-Attention for Sequence Pairs**
   - Build a model for RNA-protein interaction prediction
   - Use cross-attention between sequences
   - Implement position encoding

3. **Mini-Transformer**
   - Create a simplified transformer for sequence classification
   - Implement multiple attention layers
   - Compare with CNN and RNN approaches

**Code Starter:**
```python
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: Implement multi-head attention
        
    def forward(self, query, key, value, mask=None):
        # TODO: Implement attention mechanism
        pass

class SequenceTransformer(nn.Module):
    # TODO: Build transformer model
    pass
```

---

### **Session 8: Variational Autoencoders (VAE)**
**Source**: Learn Generative AI with PyTorch, Early chapters on VAEs

**Genomics Context**: Dimensionality reduction and data generation

**Learning Objectives:**
- Understand VAE architecture
- Implement reparameterization trick
- Generate synthetic genomic data

**Practice Exercises:**

1. **Gene Expression VAE**
   - Build a VAE for gene expression data
   - Learn latent representations
   - Visualize latent space

2. **Sequence Generation VAE**
   - Create a VAE that generates DNA sequences
   - Ensure biological validity
   - Sample from latent space

3. **Conditional VAE**
   - Implement a conditional VAE for cell-type-specific generation
   - Control generation with labels
   - Interpolate between cell types

**Code Starter:**
```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # TODO: Define encoder layers
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            # TODO: Define decoder layers
        )
        
    def encode(self, x):
        # TODO: Implement encoding
        pass
        
    def reparameterize(self, mu, logvar):
        # TODO: Implement reparameterization trick
        pass
        
    def decode(self, z):
        # TODO: Implement decoding
        pass
        
    def forward(self, x):
        # TODO: Implement full forward pass
        pass

def vae_loss(recon_x, x, mu, logvar):
    # TODO: Implement VAE loss (reconstruction + KL divergence)
    pass
```

---

### **Session 9: Generative Adversarial Networks (GANs)**
**Source**: Learn Generative AI with PyTorch, Chapters on GANs

**Genomics Context**: Generating realistic genomic sequences

**Learning Objectives:**
- Understand GAN training dynamics
- Implement generator and discriminator
- Apply to genomic data generation

**Practice Exercises:**

1. **Sequence GAN**
   - Build a GAN to generate DNA sequences
   - Ensure mode coverage
   - Evaluate quality of generated sequences

2. **Expression Profile GAN**
   - Generate realistic gene expression profiles
   - Condition on cell type or treatment
   - Implement techniques to stabilize training

3. **Wasserstein GAN**
   - Implement WGAN with gradient penalty
   - Apply to generating protein sequences
   - Compare with standard GAN

**Code Starter:**
```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # TODO: Define generator architecture
        
    def forward(self, z):
        # TODO: Generate samples from noise
        pass

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # TODO: Define discriminator architecture
        
    def forward(self, x):
        # TODO: Classify real vs fake
        pass

# Training loop
def train_gan(generator, discriminator, dataloader, epochs):
    # TODO: Implement alternating training
    pass
```

---

### **Session 10: Diffusion Models**
**Source**: Learn Generative AI with PyTorch, Advanced chapters

**Genomics Context**: State-of-the-art sequence generation

**Learning Objectives:**
- Understand diffusion process
- Implement forward and reverse diffusion
- Generate high-quality genomic sequences

**Practice Exercises:**

1. **Simple Diffusion Model**
   - Implement DDPM for 1D sequence data
   - Understand noise scheduling
   - Generate DNA sequences

2. **Conditional Diffusion**
   - Build a classifier-free guided diffusion model
   - Condition on genomic features
   - Control generation with guidance scale

3. **Protein Design with Diffusion**
   - Create a diffusion model for protein sequence generation
   - Incorporate structural constraints
   - Sample novel sequences with desired properties

**Code Starter:**
```python
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, seq_length, hidden_dim):
        super().__init__()
        # TODO: Define the noise prediction network
        
    def forward(self, x, t):
        # TODO: Predict noise at timestep t
        pass

class DiffusionProcess:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        # TODO: Define beta schedule
        
    def q_sample(self, x_0, t, noise):
        # Forward diffusion process
        # TODO: Add noise according to schedule
        pass
        
    def p_sample(self, model, x_t, t):
        # Reverse diffusion step
        # TODO: Implement denoising step
        pass
        
    def sample(self, model, shape):
        # TODO: Generate samples
        pass
```

---

## **LEVEL 4: Production & Optimization**

### **Session 11: Model Optimization**
**Source**: Deep Learning with PyTorch, Chapter 12 - "Using deployable models"

**Learning Objectives:**
- Optimize model performance
- Reduce memory footprint
- Improve inference speed

**Practice Exercises:**

1. **Mixed Precision Training**
   - Implement automatic mixed precision for a genomics model
   - Compare training speed and memory usage
   - Ensure numerical stability

2. **Model Quantization**
   - Quantize a trained sequence classifier
   - Evaluate accuracy vs size tradeoffs
   - Deploy quantized model

3. **Pruning and Distillation**
   - Prune a large model for mobile deployment
   - Distill knowledge to a smaller student model
   - Measure inference time improvements

**Code Starter:**
```python
from torch.cuda.amp import autocast, GradScaler

# Mixed precision training
scaler = GradScaler()

def train_step(model, data, target):
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### **Session 12: Large-Scale Genomics**
**Source**: Both books, scaling concepts

**Genomics Context**: Working with whole-genome datasets

**Learning Objectives:**
- Handle large genomic datasets
- Implement efficient data loading
- Distribute training across GPUs

**Practice Exercises:**

1. **Efficient Data Pipeline**
   - Create a custom Dataset for large genomic files
   - Implement efficient data loading with DataLoader
   - Use memory mapping for huge files

2. **Multi-GPU Training**
   - Implement DataParallel for sequence models
   - Use DistributedDataParallel for better performance
   - Handle batch normalization correctly

3. **Genomic Data Augmentation**
   - Implement sequence augmentation techniques
   - Random reverse complement and mutations
   - Create augmentation pipeline

**Code Starter:**
```python
import torch
from torch.utils.data import Dataset, DataLoader

class GenomicDataset(Dataset):
    def __init__(self, fasta_file, labels_file):
        # TODO: Implement efficient loading
        pass
        
    def __len__(self):
        # TODO
        pass
        
    def __getitem__(self, idx):
        # TODO: Return sequence and label
        pass

# Multi-GPU training
model = nn.DataParallel(model)
# or
from torch.nn.parallel import DistributedDataParallel
model = DistributedDataParallel(model)
```

---

## **PROJECT-BASED SESSIONS**

### **Capstone Project 1: Variant Effect Prediction**
**Combines**: Sessions 1-6

**Objective**: Build an end-to-end model to predict the functional impact of genetic variants

**Components:**
1. Data preprocessing and encoding
2. CNN + attention architecture
3. Training with transfer learning
4. Model interpretation and visualization
5. Performance evaluation on held-out test set

---

### **Capstone Project 2: Single-Cell RNA-seq Analysis**
**Combines**: Sessions 7-9

**Objective**: Create a generative model for single-cell gene expression

**Components:**
1. VAE for dimensionality reduction
2. Clustering in latent space
3. Conditional generation of cell types
4. Trajectory analysis
5. Novel cell state generation

---

### **Capstone Project 3: De Novo Sequence Design**
**Combines**: Sessions 8-10

**Objective**: Generate novel functional genomic sequences

**Components:**
1. Train diffusion model on functional sequences
2. Implement conditional generation
3. Validate generated sequences
4. Optimize for desired properties
5. Compare with real sequences

---

## 📝 Assessment Guidelines

For each session:
- **Understanding**: Can you explain the concepts in your own words?
- **Implementation**: Can you write the code without looking at solutions?
- **Adaptation**: Can you modify the code for new genomics problems?
- **Debugging**: Can you fix errors and understand why they occurred?

## 🎓 Learning Tips

1. **Code First**: Type out all examples, don't copy-paste
2. **Experiment**: Change hyperparameters and observe effects
3. **Visualize**: Plot losses, weights, and outputs
4. **Document**: Add comments explaining genomics context
5. **Validate**: Use biological knowledge to check if results make sense
6. **Iterate**: Start simple, then add complexity

## 📚 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Genomics Deep Learning Papers: AlphaFold, DeepBind, Basenji, Enformer
- Practice Datasets: UCI ML Repository, ENCODE, GTEx

---

## 🚀 Getting Started

Start with **Session 1** and complete all exercises before moving forward. Each session should take 2-4 hours to complete thoroughly. Good luck!
