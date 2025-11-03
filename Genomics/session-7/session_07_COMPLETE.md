# Session 7: Transformers and Self-Attention
**Book Ref**: Gen AI Ch. 9-11 (pages 197-278) | **Duration**: 4-5 hours

## Core Functions
```python
nn.MultiheadAttention(embed_dim, num_heads)
nn.TransformerEncoderLayer(d_model, nhead)
nn.TransformerEncoder(encoder_layer, num_layers)
```

## Exercise 1: Self-Attention from Scratch
```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V), attention_weights
```

## Exercise 2: DNA Transformer
```python
class DNATransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(d_model, 2)
    
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))
```

*Includes BERT-style pre-training for DNA*
