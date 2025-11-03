# Session 10: Diffusion Models
**Book Ref**: Gen AI diffusion sections (pages 357-360) | **Duration**: 4-5 hours

## Core Functions
```python
torch.randn_like()  # Add noise
alpha_bar  # Noise schedule
```

## Exercise 1: Forward Diffusion
```python
def forward_diffusion(x0, t, noise_schedule):
    noise = torch.randn_like(x0)
    alpha_bar = noise_schedule[t]
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    return xt, noise

# Beta schedule
betas = torch.linspace(0.0001, 0.02, 1000)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
```

## Exercise 2: Denoising Model
```python
class DenoisingModel(nn.Module):
    def __init__(self, seq_len=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len + 1, 512),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len)
        )
    
    def forward(self, x, t):
        t_emb = t.float().unsqueeze(1) / 1000
        x_t = torch.cat([x, t_emb], dim=1)
        return self.net(x_t)

# Training
for epoch in range(epochs):
    t = torch.randint(0, 1000, (batch_size,))
    xt, noise = forward_diffusion(x0, t, alpha_bar)
    predicted_noise = model(xt, t)
    loss = F.mse_loss(predicted_noise, noise)
```

*Protein sequence design application*
