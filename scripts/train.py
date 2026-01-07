import sys
import os

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.synthetic_logistics_dataset import SyntheticLogisticsDataset
from torch.utils.data import DataLoader

def load_ids(path):
    with open(path) as f:
        return [line.strip() for line in f]

DATA_ROOT = os.path.join(PROJECT_ROOT, "test_dataset", "point_cloud")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "splits")

train_ids = load_ids(os.path.join(SPLIT_DIR, "train.txt"))
val_ids   = load_ids(os.path.join(SPLIT_DIR, "val.txt"))

train_dataset = SyntheticLogisticsDataset(DATA_ROOT, train_ids)
val_dataset   = SyntheticLogisticsDataset(DATA_ROOT, val_ids)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1)

batch = next(iter(train_loader))
print("coord:", batch["coord"].shape)
print("feat:", batch["feat"].shape)
print("semantic:", batch["semantic_label"].shape)
print("instance:", batch["instance_label"].shape)