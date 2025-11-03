# Session 12: Large-Scale Genomics with PyTorch
**Book Ref**: Deep Learning PyTorch Ch. 12, 16 | **Duration**: 3-4 hours

## Core Functions
```python
torch.utils.data.Dataset
torch.utils.data.DataLoader
torch.nn.parallel.DistributedDataParallel
torch.distributed.init_process_group()
```

## Exercise 1: Custom Genomics Dataset
```python
from torch.utils.data import Dataset, DataLoader

class GenomicsDataset(Dataset):
    def __init__(self, fasta_file):
        self.sequences = []
        self.labels = []
        # Load data efficiently
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # One-hot encode
        return self.encode(seq), label

dataset = GenomicsDataset('data.fasta')
dataloader = DataLoader(dataset, batch_size=32, 
                        num_workers=4, pin_memory=True)
```

## Exercise 2: Efficient Data Loading
```python
def collate_fn(batch):
    # Custom collation for variable-length sequences
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    
    padded = torch.zeros(len(sequences), 4, max_len)
    for i, seq in enumerate(sequences):
        padded[i, :, :lengths[i]] = seq
    
    return padded, torch.tensor(labels), torch.tensor(lengths)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

## Exercise 3: Multi-GPU Training
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size):
    setup(rank, world_size)
    
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            # Train as usual
            pass

# Launch with: python -m torch.distributed.launch --nproc_per_node=4 train.py
```

*Whole-genome processing pipeline included*
