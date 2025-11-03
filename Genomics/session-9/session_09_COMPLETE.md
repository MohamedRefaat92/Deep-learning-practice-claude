# Session 9: Generative Adversarial Networks (GANs)
**Book Ref**: Gen AI Ch. 4-6 (pages 67-148) | **Duration**: 3-4 hours

## Core Functions
```python
nn.BCEWithLogitsLoss()  # For discriminator
torch.randn()  # Sample noise
```

## Exercise 1: Basic GAN
```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Training loop
for epoch in range(epochs):
    # Train Discriminator
    z = torch.randn(batch_size, latent_dim)
    fake = generator(z)
    d_loss = criterion(discriminator(real), ones) +              criterion(discriminator(fake.detach()), zeros)
    
    # Train Generator
    g_loss = criterion(discriminator(fake), ones)
```

## Exercise 2: Conditional GAN for Sequences
```python
class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        # Concatenate noise and label embedding
```

*WGAN-GP implementation included*
