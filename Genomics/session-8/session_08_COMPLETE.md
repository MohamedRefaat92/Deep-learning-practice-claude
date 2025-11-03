# Session 8: Variational Autoencoders (VAEs)
**Book Ref**: Gen AI Ch. 3 (pages 47-66) | **Duration**: 3-4 hours

## Core Functions
```python
torch.distributions.Normal(mean, std)
F.binary_cross_entropy()
kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

## Exercise 1: Simple VAE
```python
class VAE(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

## Exercise 2: scRNA-seq VAE
```python
# For single-cell RNA-seq dimensionality reduction
class scRNAVAE(VAE):
    def __init__(self, n_genes=2000, latent_dim=10):
        super().__init__(n_genes, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

*Complete scRNA-seq analysis pipeline*
