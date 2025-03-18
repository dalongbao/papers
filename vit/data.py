from torch.utils.data import DataLoader, Dataset
from utils import get_dataset

batch_size=32
shuffle=True
num_workers=2
training_data, test_data = get_dataset('imagenet')
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# At the start of training
print(f"Number of training samples: {len(training_data)}")
print(f"Number of test samples: {len(test_data)}")

# Check a batch
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Label range: {y.min().item()} to {y.max().item()}")
    print(f"Sample labels: {y[:10]}")
    break
