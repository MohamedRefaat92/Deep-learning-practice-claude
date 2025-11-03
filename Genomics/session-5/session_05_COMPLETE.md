# Session 5: Recurrent Neural Networks for Sequences
**Book Ref**: Deep Learning PyTorch Ch. 9, Gen AI Ch. 8 | **Duration**: 3-4 hours

## Core Functions
```python
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
nn.GRU(input_size, hidden_size, num_layers)
nn.utils.rnn.pack_padded_sequence()
nn.utils.rnn.pad_packed_sequence()
```

## Exercise 1: Basic LSTM
```python
import torch
import torch.nn as nn

class ProteinLSTM(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 3 secondary structures
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h, c) = self.lstm(x)
        return self.fc(lstm_out)

model = ProteinLSTM()
```

## Exercise 2: Bidirectional LSTM
```python
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 1)  # 128*2 for bidirectional
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(lstm_out[:, -1, :]))
```

## Exercise 3: Variable Length Sequences
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def handle_variable_length(sequences, lengths):
    packed = pack_padded_sequence(sequences, lengths, 
                                   batch_first=True, enforce_sorted=False)
    lstm_out, _ = model.lstm(packed)
    unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
    return unpacked
```

*Complete exercises in notebook format*
