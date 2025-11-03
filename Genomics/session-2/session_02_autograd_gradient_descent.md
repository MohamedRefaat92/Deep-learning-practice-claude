# Practice Session 2: Autograd and Gradient Descent
## PyTorch for Genomics - Learning from Data

**Source**: Deep Learning with PyTorch (Second Edition), Chapter 5 - "The mechanics of learning"

**Duration**: 2-3 hours  
**Difficulty**: Beginner to Intermediate  
**Prerequisites**: Completion of Session 1, understanding of basic calculus concepts

---

## 🎯 Learning Objectives

By the end of this session, you will:
1. Understand how PyTorch's automatic differentiation (autograd) works
2. Implement gradient descent from scratch and with PyTorch optimizers
3. Build complete training loops for genomics models
4. Properly split data into training and validation sets
5. Recognize and prevent overfitting
6. Compare different optimization strategies (SGD vs Adam)
7. Visualize learning curves and diagnose training issues

---

## 📚 Theory Review

### What is Learning?

**Learning** in machine learning means adjusting model parameters to minimize a loss function. Think of it as:
1. **Model**: A function that makes predictions (e.g., predicting if a variant is pathogenic)
2. **Loss**: A measure of how wrong the predictions are
3. **Gradient**: The direction to adjust parameters to reduce loss
4. **Optimizer**: The strategy for updating parameters using gradients

### The Learning Process

```
1. Forward Pass:  inputs → model → predictions
2. Loss Calculation: compare predictions with truth
3. Backward Pass: compute gradients (how to improve)
4. Update: adjust parameters to reduce loss
5. Repeat until loss is minimized
```

### Key Concepts

**Gradient Descent**: Move parameters in the direction that reduces loss
- Learning Rate: How big each step should be
- Too large: Training unstable, loss oscillates
- Too small: Training too slow, may get stuck

**Autograd**: PyTorch automatically computes derivatives
- Tracks operations in a computation graph
- Computes gradients via backpropagation
- Essential for training neural networks

---

## 🧪 Exercise 1: Manual Gradient Descent for Gene Expression

### Part A: Problem Setup

We'll predict gene expression levels from a simple linear model based on transcription factor binding scores.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Simulate genomics data
# True relationship: expression = 2.5 * TF_binding + 1.0 + noise
torch.manual_seed(42)

# Transcription factor binding scores (normalized 0-1)
tf_binding = torch.linspace(0, 1, 50)

# True gene expression levels (with noise)
true_weight = 2.5
true_bias = 1.0
gene_expression = true_weight * tf_binding + true_bias + torch.randn(50) * 0.2

print(f"TF binding scores shape: {tf_binding.shape}")
print(f"Gene expression shape: {gene_expression.shape}")
print(f"\nFirst 5 samples:")
print(f"TF binding: {tf_binding[:5]}")
print(f"Expression: {gene_expression[:5]}")

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(tf_binding, gene_expression, alpha=0.6, label='Data')
plt.xlabel('TF Binding Score')
plt.ylabel('Gene Expression Level')
plt.title('Transcription Factor Binding vs Gene Expression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Part B: Define Model and Loss

```python
def linear_model(x, weight, bias):
    """
    Simple linear model: y = w * x + b
    
    Args:
        x: Input tensor (TF binding scores)
        weight: Model parameter (slope)
        bias: Model parameter (intercept)
    
    Returns:
        Predictions
    """
    return weight * x + bias

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss function.
    
    Args:
        predictions: Model outputs
        targets: True values
        
    Returns:
        Average squared difference
    """
    squared_diff = (predictions - targets) ** 2
    return squared_diff.mean()

# Initialize parameters randomly
weight = torch.randn(1)
bias = torch.randn(1)

print(f"Initial weight: {weight.item():.4f}")
print(f"Initial bias: {bias.item():.4f}")

# Make initial predictions
initial_predictions = linear_model(tf_binding, weight, bias)
initial_loss = mse_loss(initial_predictions, gene_expression)

print(f"\nInitial loss: {initial_loss.item():.4f}")
```

### Part C: Manual Gradient Computation

```python
def compute_gradients(x, y, weight, bias):
    """
    Manually compute gradients of MSE loss.
    
    For loss L = mean((y_pred - y_true)^2)
    Where y_pred = w*x + b
    
    Gradients:
    dL/dw = mean(2 * (y_pred - y_true) * x)
    dL/db = mean(2 * (y_pred - y_true))
    
    Args:
        x: Input features
        y: True targets
        weight: Current weight parameter
        bias: Current bias parameter
        
    Returns:
        Tuple of (weight_gradient, bias_gradient)
    """
    # Task 1.1: Implement manual gradient computation
    # YOUR CODE HERE
    
    # Forward pass
    y_pred = linear_model(x, weight, bias)
    
    # Compute error
    error = y_pred - y
    
    # Compute gradients
    n = x.shape[0]
    weight_grad = (2 / n) * (error * x).sum()
    bias_grad = (2 / n) * error.sum()
    
    return weight_grad, bias_grad

# Test gradient computation
w_grad, b_grad = compute_gradients(tf_binding, gene_expression, weight, bias)
print(f"Weight gradient: {w_grad.item():.4f}")
print(f"Bias gradient: {b_grad.item():.4f}")
```

### Part D: Training Loop with Manual Gradients

```python
def train_manual_gradient_descent(x, y, epochs=100, learning_rate=0.1):
    """
    Train linear model using manually computed gradients.
    
    Args:
        x: Input features
        y: Targets
        epochs: Number of training iterations
        learning_rate: Step size for parameter updates
        
    Returns:
        Tuple of (final_weight, final_bias, loss_history)
    """
    # Task 1.2: Implement training loop
    # YOUR CODE HERE
    
    # Initialize parameters
    weight = torch.randn(1)
    bias = torch.randn(1)
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = linear_model(x, weight, bias)
        loss = mse_loss(predictions, y)
        loss_history.append(loss.item())
        
        # Compute gradients manually
        w_grad, b_grad = compute_gradients(x, y, weight, bias)
        
        # Update parameters (gradient descent step)
        weight = weight - learning_rate * w_grad
        bias = bias - learning_rate * b_grad
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, "
                  f"Weight = {weight.item():.4f}, Bias = {bias.item():.4f}")
    
    return weight, bias, loss_history

# Train the model
print("Training with manual gradient descent:\n")
final_weight, final_bias, loss_history = train_manual_gradient_descent(
    tf_binding, gene_expression, epochs=100, learning_rate=0.1
)

print(f"\nFinal parameters:")
print(f"Weight: {final_weight.item():.4f} (true: {true_weight})")
print(f"Bias: {final_bias.item():.4f} (true: {true_bias})")

# Visualize training progress
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🧪 Exercise 2: PyTorch Autograd

### Part A: Understanding requires_grad

```python
import torch

# Task 2.1: Experiment with requires_grad
# YOUR CODE HERE

# Create tensors with and without gradient tracking
x_no_grad = torch.tensor([1.0, 2.0, 3.0])
x_with_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

print("Tensor without gradient tracking:")
print(f"  Value: {x_no_grad}")
print(f"  requires_grad: {x_no_grad.requires_grad}")
print(f"  grad: {x_no_grad.grad}")

print("\nTensor with gradient tracking:")
print(f"  Value: {x_with_grad}")
print(f"  requires_grad: {x_with_grad.requires_grad}")
print(f"  grad: {x_with_grad.grad}")

# Perform operation
y = (x_with_grad ** 2).sum()
print(f"\nAfter operation y = sum(x^2):")
print(f"  y value: {y.item()}")
print(f"  y requires_grad: {y.requires_grad}")

# Compute gradients
y.backward()

print(f"\nAfter calling y.backward():")
print(f"  x gradient (dy/dx): {x_with_grad.grad}")
print(f"  Expected: 2*x = {2 * x_with_grad.detach()}")
```

### Part B: Autograd for Linear Model

```python
def train_with_autograd(x, y, epochs=100, learning_rate=0.1):
    """
    Train linear model using PyTorch autograd.
    
    Args:
        x: Input features
        y: Targets
        epochs: Number of training iterations
        learning_rate: Step size for parameter updates
        
    Returns:
        Tuple of (final_weight, final_bias, loss_history)
    """
    # Task 2.2: Implement training with autograd
    # YOUR CODE HERE
    
    # Initialize parameters with gradient tracking
    weight = torch.randn(1, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = linear_model(x, weight, bias)
        loss = mse_loss(predictions, y)
        loss_history.append(loss.item())
        
        # CRITICAL: Zero gradients from previous iteration
        if weight.grad is not None:
            weight.grad.zero_()
        if bias.grad is not None:
            bias.grad.zero_()
        
        # Backward pass - autograd computes gradients!
        loss.backward()
        
        # Update parameters (no gradient tracking for this operation)
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, "
                  f"Weight = {weight.item():.4f}, Bias = {bias.item():.4f}")
    
    return weight, bias, loss_history

# Train the model
print("Training with PyTorch autograd:\n")
final_weight_auto, final_bias_auto, loss_history_auto = train_with_autograd(
    tf_binding, gene_expression, epochs=100, learning_rate=0.1
)

print(f"\nFinal parameters:")
print(f"Weight: {final_weight_auto.item():.4f} (true: {true_weight})")
print(f"Bias: {final_bias_auto.item():.4f} (true: {true_bias})")

# Compare with manual gradients
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Manual Gradients', alpha=0.7)
plt.plot(loss_history_auto, label='Autograd', alpha=0.7, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss: Manual vs Autograd')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Part C: Understanding the Computation Graph

```python
# Task 2.3: Visualize computation graph
# YOUR CODE HERE

# Create simple computation
x = torch.tensor([2.0], requires_grad=True)
a = x * 3
b = a + 5
c = b ** 2
loss = c.mean()

print("Computation graph:")
print(f"x = {x.item()}")
print(f"a = x * 3 = {a.item()}")
print(f"b = a + 5 = {b.item()}")
print(f"c = b^2 = {c.item()}")
print(f"loss = mean(c) = {loss.item()}")

# Compute gradient
loss.backward()

print(f"\nGradient dl/dx = {x.grad.item()}")

# Manual verification using chain rule
# dl/dx = dl/dc * dc/db * db/da * da/dx
# dl/dc = 1 (derivative of mean)
# dc/db = 2*b
# db/da = 1
# da/dx = 3
manual_grad = 1.0 * 2 * b.item() * 1.0 * 3.0
print(f"Manual calculation: {manual_grad}")
```

---

## 🧪 Exercise 3: PyTorch Optimizers

### Part A: SGD Optimizer

```python
import torch.optim as optim

def train_with_optimizer(x, y, optimizer_class, epochs=100, learning_rate=0.1, **optimizer_kwargs):
    """
    Train linear model using PyTorch optimizer.
    
    Args:
        x: Input features
        y: Targets
        optimizer_class: Optimizer class (e.g., optim.SGD, optim.Adam)
        epochs: Number of training iterations
        learning_rate: Learning rate
        **optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        Tuple of (parameters, loss_history)
    """
    # Task 3.1: Implement training with PyTorch optimizer
    # YOUR CODE HERE
    
    # Initialize parameters
    weight = torch.randn(1, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    
    # Create optimizer
    optimizer = optimizer_class([weight, bias], lr=learning_rate, **optimizer_kwargs)
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = linear_model(x, weight, bias)
        loss = mse_loss(predictions, y)
        loss_history.append(loss.item())
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, "
                  f"Weight = {weight.item():.4f}, Bias = {bias.item():.4f}")
    
    return (weight, bias), loss_history

# Train with SGD
print("Training with SGD optimizer:\n")
params_sgd, loss_sgd = train_with_optimizer(
    tf_binding, gene_expression, 
    optim.SGD, 
    epochs=100, 
    learning_rate=0.1
)

weight_sgd, bias_sgd = params_sgd
print(f"\nFinal SGD parameters:")
print(f"Weight: {weight_sgd.item():.4f} (true: {true_weight})")
print(f"Bias: {bias_sgd.item():.4f} (true: {true_bias})")
```

### Part B: Adam Optimizer

```python
# Task 3.2: Train with Adam optimizer
# YOUR CODE HERE

print("\nTraining with Adam optimizer:\n")
params_adam, loss_adam = train_with_optimizer(
    tf_binding, gene_expression, 
    optim.Adam, 
    epochs=100, 
    learning_rate=0.1
)

weight_adam, bias_adam = params_adam
print(f"\nFinal Adam parameters:")
print(f"Weight: {weight_adam.item():.4f} (true: {true_weight})")
print(f"Bias: {bias_adam.item():.4f} (true: {true_bias})")
```

### Part C: Comparing Optimizers

```python
# Task 3.3: Compare different optimizers
# YOUR CODE HERE

# Test multiple learning rates with SGD
learning_rates = [0.01, 0.1, 0.5]
plt.figure(figsize=(15, 5))

for i, lr in enumerate(learning_rates, 1):
    _, loss_history = train_with_optimizer(
        tf_binding, gene_expression, 
        optim.SGD, 
        epochs=100, 
        learning_rate=lr,
        verbose=False
    )
    
    plt.subplot(1, 3, i)
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'SGD with lr={lr}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare optimizers at same learning rate
plt.figure(figsize=(10, 6))

optimizers = [
    (optim.SGD, 'SGD'),
    (optim.Adam, 'Adam'),
    (optim.RMSprop, 'RMSprop')
]

for opt_class, name in optimizers:
    _, loss_history = train_with_optimizer(
        tf_binding, gene_expression, 
        opt_class, 
        epochs=100, 
        learning_rate=0.1,
        verbose=False
    )
    plt.plot(loss_history, label=name, alpha=0.7)

plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🧪 Exercise 4: Variant Classification (Binary)

Build a classifier to predict if genetic variants are pathogenic or benign.

### Part A: Generate Synthetic Variant Data

```python
import torch
import torch.nn.functional as F

def generate_variant_data(n_samples=1000, seed=42):
    """
    Generate synthetic variant data for binary classification.
    
    Features:
        - Conservation score (0-1)
        - Allele frequency (0-1)
        - CADD score (0-40)
        - PolyPhen score (0-1)
        - SIFT score (0-1)
    
    Labels:
        - 0: Benign
        - 1: Pathogenic
    """
    torch.manual_seed(seed)
    
    # Pathogenic variants (label=1)
    n_pathogenic = n_samples // 2
    pathogenic_features = torch.tensor([
        torch.randn(n_pathogenic) * 0.1 + 0.9,  # High conservation
        torch.randn(n_pathogenic) * 0.05 + 0.02,  # Low frequency
        torch.randn(n_pathogenic) * 5 + 25,  # High CADD
        torch.randn(n_pathogenic) * 0.1 + 0.8,  # High PolyPhen
        torch.randn(n_pathogenic) * 0.1 + 0.1,  # Low SIFT (damaging)
    ]).T
    pathogenic_labels = torch.ones(n_pathogenic)
    
    # Benign variants (label=0)
    n_benign = n_samples - n_pathogenic
    benign_features = torch.tensor([
        torch.randn(n_benign) * 0.1 + 0.3,  # Low conservation
        torch.randn(n_benign) * 0.1 + 0.5,  # High frequency
        torch.randn(n_benign) * 3 + 10,  # Low CADD
        torch.randn(n_benign) * 0.1 + 0.2,  # Low PolyPhen
        torch.randn(n_benign) * 0.1 + 0.8,  # High SIFT (tolerated)
    ]).T
    benign_labels = torch.zeros(n_benign)
    
    # Combine and shuffle
    features = torch.cat([pathogenic_features, benign_features])
    labels = torch.cat([pathogenic_labels, benign_labels])
    
    # Normalize features to [0, 1] range
    features = (features - features.min(dim=0)[0]) / (features.max(dim=0)[0] - features.min(dim=0)[0])
    
    # Shuffle
    indices = torch.randperm(n_samples)
    features = features[indices]
    labels = labels[indices]
    
    return features, labels

# Generate data
features, labels = generate_variant_data(n_samples=1000)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"\nFeature names: Conservation, Frequency, CADD, PolyPhen, SIFT")
print(f"First 3 samples:")
print(features[:3])
print(f"Labels: {labels[:3]}")
print(f"\nClass distribution:")
print(f"Benign (0): {(labels == 0).sum().item()}")
print(f"Pathogenic (1): {(labels == 1).sum().item()}")
```

### Part B: Logistic Regression Model

```python
def logistic_model(x, weights, bias):
    """
    Logistic regression: sigmoid(w * x + b)
    
    Args:
        x: Input features (n_samples, n_features)
        weights: Model weights (n_features,)
        bias: Model bias (scalar)
        
    Returns:
        Probabilities (n_samples,)
    """
    # Task 4.1: Implement logistic regression
    # YOUR CODE HERE
    logits = x @ weights + bias
    probabilities = torch.sigmoid(logits)
    return probabilities

def binary_cross_entropy_loss(predictions, targets):
    """
    Binary cross-entropy loss.
    
    Loss = -mean(y*log(p) + (1-y)*log(1-p))
    
    Args:
        predictions: Predicted probabilities (0-1)
        targets: True labels (0 or 1)
        
    Returns:
        Loss value
    """
    # Task 4.2: Implement BCE loss
    # YOUR CODE HERE
    epsilon = 1e-7  # For numerical stability
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    loss = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    return loss.mean()

# Initialize parameters
n_features = features.shape[1]
weights = torch.randn(n_features, requires_grad=True) * 0.1
bias = torch.zeros(1, requires_grad=True)

print(f"Model initialized:")
print(f"  Weights shape: {weights.shape}")
print(f"  Bias shape: {bias.shape}")

# Test forward pass
test_probs = logistic_model(features[:5], weights, bias)
print(f"\nTest predictions: {test_probs}")
test_loss = binary_cross_entropy_loss(test_probs, labels[:5])
print(f"Test loss: {test_loss.item():.4f}")
```

### Part C: Training the Classifier

```python
def train_classifier(features, labels, epochs=100, learning_rate=0.1):
    """
    Train logistic regression classifier.
    
    Args:
        features: Input features
        labels: True labels
        epochs: Number of training iterations
        learning_rate: Learning rate
        
    Returns:
        Tuple of (weights, bias, loss_history, accuracy_history)
    """
    # Task 4.3: Implement classifier training
    # YOUR CODE HERE
    
    n_features = features.shape[1]
    weights = torch.randn(n_features, requires_grad=True) * 0.1
    bias = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.Adam([weights, bias], lr=learning_rate)
    
    loss_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = logistic_model(features, weights, bias)
        loss = binary_cross_entropy_loss(predictions, labels)
        
        # Calculate accuracy
        predicted_classes = (predictions > 0.5).float()
        accuracy = (predicted_classes == labels).float().mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        loss_history.append(loss.item())
        accuracy_history.append(accuracy.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, "
                  f"Accuracy = {accuracy.item():.4f}")
    
    return weights, bias, loss_history, accuracy_history

# Train classifier
print("Training variant classifier:\n")
weights, bias, loss_hist, acc_hist = train_classifier(
    features, labels, epochs=100, learning_rate=0.1
)

# Visualize training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(loss_hist)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (BCE)')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(acc_hist)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal accuracy: {acc_hist[-1]:.4f}")
```

---

## 🧪 Exercise 5: Training and Validation Split

### Part A: Split the Data

```python
def split_data(features, labels, train_ratio=0.8, seed=42):
    """
    Split data into training and validation sets.
    
    Args:
        features: Input features
        labels: Labels
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels)
    """
    # Task 5.1: Implement data splitting
    # YOUR CODE HERE
    
    torch.manual_seed(seed)
    
    n_samples = features.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # Random permutation
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Split data
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    return train_features, train_labels, val_features, val_labels

# Split the variant data
train_feats, train_labs, val_feats, val_labs = split_data(features, labels)

print(f"Training set:")
print(f"  Features: {train_feats.shape}")
print(f"  Labels: {train_labs.shape}")
print(f"  Pathogenic: {(train_labs == 1).sum().item()}")

print(f"\nValidation set:")
print(f"  Features: {val_feats.shape}")
print(f"  Labels: {val_labs.shape}")
print(f"  Pathogenic: {(val_labs == 1).sum().item()}")
```

### Part B: Training with Validation

```python
def train_with_validation(train_features, train_labels, val_features, val_labels, 
                          epochs=100, learning_rate=0.1):
    """
    Train classifier with separate validation monitoring.
    
    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        epochs: Number of training iterations
        learning_rate: Learning rate
        
    Returns:
        Tuple of (weights, bias, train_history, val_history)
    """
    # Task 5.2: Implement training with validation
    # YOUR CODE HERE
    
    n_features = train_features.shape[1]
    weights = torch.randn(n_features, requires_grad=True) * 0.1
    bias = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.Adam([weights, bias], lr=learning_rate)
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        # Training
        train_preds = logistic_model(train_features, weights, bias)
        train_loss = binary_cross_entropy_loss(train_preds, train_labels)
        train_acc = ((train_preds > 0.5).float() == train_labels).float().mean()
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation (no gradient computation)
        with torch.no_grad():
            val_preds = logistic_model(val_features, weights, bias)
            val_loss = binary_cross_entropy_loss(val_preds, val_labels)
            val_acc = ((val_preds > 0.5).float() == val_labels).float().mean()
        
        # Record metrics
        train_loss_history.append(train_loss.item())
        train_acc_history.append(train_acc.item())
        val_loss_history.append(val_loss.item())
        val_acc_history.append(val_acc.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss = {train_loss.item():.4f}, Train Acc = {train_acc.item():.4f} | "
                  f"Val Loss = {val_loss.item():.4f}, Val Acc = {val_acc.item():.4f}")
    
    return weights, bias, (train_loss_history, train_acc_history), (val_loss_history, val_acc_history)

# Train with validation
print("Training with validation monitoring:\n")
weights, bias, train_hist, val_hist = train_with_validation(
    train_feats, train_labs, val_feats, val_labs, 
    epochs=100, learning_rate=0.1
)

train_loss, train_acc = train_hist
val_loss, val_acc = val_hist
```

### Part C: Visualizing Training vs Validation

```python
# Task 5.3: Create comprehensive visualization
# YOUR CODE HERE

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Loss comparison
ax1.plot(train_loss, label='Training', alpha=0.7)
ax1.plot(val_loss, label='Validation', alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (BCE)')
ax1.set_title('Training vs Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy comparison
ax2.plot(train_acc, label='Training', alpha=0.7)
ax2.plot(val_acc, label='Validation', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training vs Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Loss difference (overfitting indicator)
loss_diff = [t - v for t, v in zip(train_loss, val_loss)]
ax3.plot(loss_diff)
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Train Loss - Val Loss')
ax3.set_title('Overfitting Indicator (should stay near 0)')
ax3.grid(True, alpha=0.3)

# Final comparison
metrics = ['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']
values = [train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]]
colors = ['blue', 'orange', 'blue', 'orange']
ax4.bar(metrics, values, color=colors, alpha=0.7)
ax4.set_ylabel('Value')
ax4.set_title('Final Metrics Comparison')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\nFinal Results:")
print(f"Training   - Loss: {train_loss[-1]:.4f}, Accuracy: {train_acc[-1]:.4f}")
print(f"Validation - Loss: {val_loss[-1]:.4f}, Accuracy: {val_acc[-1]:.4f}")
print(f"Gap        - Loss: {abs(train_loss[-1] - val_loss[-1]):.4f}, "
      f"Accuracy: {abs(train_acc[-1] - val_acc[-1]):.4f}")
```

---

## 🧪 Exercise 6: Detecting and Preventing Overfitting

### Part A: Create Overfitting Scenario

```python
def train_with_capacity_control(train_feats, train_labs, val_feats, val_labs, 
                                 n_params_multiplier=1, epochs=200, learning_rate=0.1):
    """
    Train model with controlled capacity to demonstrate overfitting.
    
    Args:
        train_feats: Training features
        train_labs: Training labels
        val_feats: Validation features
        val_labs: Validation labels
        n_params_multiplier: Scale number of parameters (1 = normal, >1 = more capacity)
        epochs: Training iterations
        learning_rate: Learning rate
        
    Returns:
        Training and validation histories
    """
    # Task 6.1: Implement model with variable capacity
    # YOUR CODE HERE
    
    n_features = train_feats.shape[1]
    n_params = int(n_features * n_params_multiplier)
    
    # Add polynomial features to increase model capacity
    if n_params_multiplier > 1:
        # Square features for more expressive power
        train_feats_expanded = torch.cat([train_feats, train_feats ** 2], dim=1)
        val_feats_expanded = torch.cat([val_feats, val_feats ** 2], dim=1)
    else:
        train_feats_expanded = train_feats
        val_feats_expanded = val_feats
    
    weights = torch.randn(train_feats_expanded.shape[1], requires_grad=True) * 0.1
    bias = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.Adam([weights, bias], lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        train_preds = logistic_model(train_feats_expanded, weights, bias)
        train_loss = binary_cross_entropy_loss(train_preds, train_labs)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_preds = logistic_model(val_feats_expanded, weights, bias)
            val_loss = binary_cross_entropy_loss(val_preds, val_labs)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
    
    return train_losses, val_losses

# Compare different model capacities
print("Comparing model capacities:\n")

capacities = [1, 2, 5]  # 1x, 2x, 5x parameters
plt.figure(figsize=(15, 10))

for idx, capacity in enumerate(capacities, 1):
    print(f"Training with {capacity}x capacity...")
    train_losses, val_losses = train_with_capacity_control(
        train_feats, train_labs, val_feats, val_labs,
        n_params_multiplier=capacity, epochs=200, learning_rate=0.05
    )
    
    plt.subplot(2, 2, idx)
    plt.plot(train_losses, label='Training', alpha=0.7)
    plt.plot(val_losses, label='Validation', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Model Capacity: {capacity}x')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if capacity > 1 and val_losses[-1] > val_losses[len(val_losses)//2]:
        plt.axvline(x=np.argmin(val_losses), color='r', linestyle='--', 
                   label='Best Val Loss', alpha=0.5)

plt.subplot(2, 2, 4)
plt.text(0.5, 0.5, 'Signs of Overfitting:\n\n'
                   '1. Validation loss increases\n'
                   '   while training decreases\n\n'
                   '2. Large gap between\n'
                   '   train and val loss\n\n'
                   '3. Best validation occurs\n'
                   '   early in training',
         ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')

plt.tight_layout()
plt.show()
```

### Part B: Early Stopping

```python
def train_with_early_stopping(train_feats, train_labs, val_feats, val_labs,
                               epochs=200, patience=10, learning_rate=0.1):
    """
    Train with early stopping to prevent overfitting.
    
    Args:
        train_feats: Training features
        train_labs: Training labels
        val_feats: Validation features
        val_labs: Validation labels
        epochs: Maximum training iterations
        patience: Number of epochs to wait for improvement
        learning_rate: Learning rate
        
    Returns:
        Best model parameters and training history
    """
    # Task 6.2: Implement early stopping
    # YOUR CODE HERE
    
    n_features = train_feats.shape[1]
    weights = torch.randn(n_features, requires_grad=True) * 0.1
    bias = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.Adam([weights, bias], lr=learning_rate)
    
    best_val_loss = float('inf')
    best_weights = None
    best_bias = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        train_preds = logistic_model(train_feats, weights, bias)
        train_loss = binary_cross_entropy_loss(train_preds, train_labs)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_preds = logistic_model(val_feats, weights, bias)
            val_loss = binary_cross_entropy_loss(val_preds, val_labs)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_weights = weights.clone().detach()
            best_bias = bias.clone().detach()
            patience_counter = 0
            print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    return best_weights, best_bias, train_losses, val_losses, epoch+1

# Train with early stopping
print("Training with early stopping:\n")
best_w, best_b, train_l, val_l, stopped_epoch = train_with_early_stopping(
    train_feats, train_labs, val_feats, val_labs,
    epochs=200, patience=15, learning_rate=0.1
)

# Visualize early stopping
plt.figure(figsize=(10, 6))
plt.plot(train_l, label='Training', alpha=0.7)
plt.plot(val_l, label='Validation', alpha=0.7)
plt.axvline(x=stopped_epoch, color='r', linestyle='--', label='Early Stop', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping in Action')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🎯 Challenge Problems

### Challenge 1: Multi-Class Gene Expression Classification

Extend the binary classifier to predict gene expression levels in multiple categories (low, medium, high).

```python
def generate_multiclass_data(n_samples=1000, n_classes=3, seed=42):
    """
    Generate synthetic multi-class gene expression data.
    
    Classes: 0=Low, 1=Medium, 2=High expression
    """
    # Task: Implement multi-class data generation
    # YOUR CODE HERE
    
    torch.manual_seed(seed)
    
    features_list = []
    labels_list = []
    
    samples_per_class = n_samples // n_classes
    
    for class_id in range(n_classes):
        # Each class has distinct feature distributions
        class_features = torch.randn(samples_per_class, 5) + class_id * 2
        class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
        
        features_list.append(class_features)
        labels_list.append(class_labels)
    
    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    
    # Shuffle
    indices = torch.randperm(n_samples)
    return features[indices], labels[indices]

# Generate multi-class data
multi_feats, multi_labels = generate_multiclass_data()

print(f"Multi-class data:")
print(f"  Features: {multi_feats.shape}")
print(f"  Labels: {multi_labels.shape}")
print(f"  Classes: {multi_labels.unique()}")
print(f"\nClass distribution:")
for c in range(3):
    print(f"  Class {c}: {(multi_labels == c).sum().item()}")

# Hint: Use F.cross_entropy() loss and F.softmax() for predictions
```

### Challenge 2: Learning Rate Scheduling

Implement learning rate scheduling to improve convergence.

```python
def train_with_lr_schedule(train_feats, train_labs, val_feats, val_labs,
                            epochs=100, initial_lr=0.1, lr_decay=0.95):
    """
    Train with exponentially decaying learning rate.
    
    Args:
        train_feats: Training features
        train_labs: Training labels  
        val_feats: Validation features
        val_labs: Validation labels
        epochs: Training iterations
        initial_lr: Starting learning rate
        lr_decay: Multiplicative decay factor per epoch
        
    Returns:
        Training history and learning rate schedule
    """
    # Task: Implement learning rate scheduling
    # YOUR CODE HERE
    
    n_features = train_feats.shape[1]
    weights = torch.randn(n_features, requires_grad=True) * 0.1
    bias = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.SGD([weights, bias], lr=initial_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(epochs):
        # Training
        train_preds = logistic_model(train_feats, weights, bias)
        train_loss = binary_cross_entropy_loss(train_preds, train_labs)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_preds = logistic_model(val_feats, weights, bias)
            val_loss = binary_cross_entropy_loss(val_preds, val_labs)
        
        # Record metrics
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: LR = {learning_rates[-1]:.6f}, "
                  f"Train Loss = {train_loss.item():.4f}, "
                  f"Val Loss = {val_loss.item():.4f}")
    
    return train_losses, val_losses, learning_rates

# Train with LR scheduling
print("Training with learning rate scheduling:\n")
train_l, val_l, lrs = train_with_lr_schedule(
    train_feats, train_labs, val_feats, val_labs,
    epochs=100, initial_lr=0.5, lr_decay=0.97
)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_l, label='Training')
ax1.plot(val_l, label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss with LR Scheduling')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(lrs)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## ✅ Self-Assessment

Before moving to Session 3, ensure you can:

- [ ] Explain how autograd builds computation graphs
- [ ] Implement gradient descent manually
- [ ] Use PyTorch optimizers (SGD, Adam, RMSprop)
- [ ] Build complete training loops with proper gradient handling
- [ ] Split data into training and validation sets
- [ ] Monitor training to detect overfitting
- [ ] Implement early stopping
- [ ] Compare different optimizers and learning rates
- [ ] Use `torch.no_grad()` appropriately
- [ ] Debug common gradient-related issues

---

## 📝 Additional Practice Ideas

1. **Experiment with momentum**: Add momentum to SGD and compare convergence
2. **Batch training**: Modify code to use mini-batches instead of full batch
3. **Weight decay**: Add L2 regularization to prevent overfitting
4. **Gradient clipping**: Implement gradient clipping for stable training
5. **Custom loss functions**: Create domain-specific loss for genomics
6. **Visualization**: Plot decision boundaries for 2D variant data
7. **Real data**: Download variant data from ClinVar and train a real classifier

---

## 🚀 Next Steps

Once you're comfortable with these exercises, move on to:
- **Session 3**: Building Neural Networks with `nn.Module`
- Try implementing batch training (process data in chunks)
- Experiment with different architectures and hyperparameters

---

## 📚 Key Takeaways

### The Training Recipe
```python
# 1. Initialize parameters
params = torch.randn(..., requires_grad=True)

# 2. Create optimizer
optimizer = optim.Adam([params], lr=0.1)

# 3. Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model(inputs, params)
    loss = loss_function(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update parameters
```

### Common Pitfalls to Avoid
1. **Forgetting `zero_grad()`**: Gradients accumulate!
2. **Wrong `requires_grad`**: Parameters must have `requires_grad=True`
3. **Training without validation**: Always monitor validation loss
4. **Ignoring overfitting**: Stop when validation loss increases
5. **Bad learning rate**: Too high = unstable, too low = slow

---

## 📖 Additional Resources

- PyTorch Autograd Tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Understanding Optimizers: https://pytorch.org/docs/stable/optim.html
- Loss Functions Guide: https://pytorch.org/docs/stable/nn.html#loss-functions
- Gradient Descent Visualization: https://distill.pub/2017/momentum/

Good luck with your practice!
