# Session 11: Model Optimization and Deployment
**Book Ref**: Deep Learning PyTorch Ch. 12, 15 (pages 371-415) | **Duration**: 3-4 hours

## Core Functions
```python
torch.cuda.amp.autocast()  # Mixed precision
torch.quantization.quantize_dynamic()  # Quantization
torch.jit.script()  # JIT compilation
```

## Exercise 1: Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Exercise 2: Model Quantization
```python
# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Compare sizes
original_size = os.path.getsize('model.pth') / 1e6
quantized_size = os.path.getsize('quantized_model.pth') / 1e6
print(f"Size reduction: {original_size:.1f}MB -> {quantized_size:.1f}MB")
```

## Exercise 3: Model Pruning
```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Make pruning permanent
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')
```

*ONNX export and deployment included*
