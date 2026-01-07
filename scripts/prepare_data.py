import os
import random

DATA_ROOT = (r"E:\softgroup_project\test_dataset\point_cloud")

# 1. Find all scenes
scene_ids = sorted([
    f.split("_pointcloud.npy")[0]
    for f in os.listdir(DATA_ROOT)
    if f.endswith("_pointcloud.npy")
])

# 2. Split
random.seed(42)
random.shuffle(scene_ids)

n = len(scene_ids)
train_ids = scene_ids[:int(0.8 * n)]
val_ids   = scene_ids[int(0.8 * n):int(0.9 * n)]
test_ids  = scene_ids[int(0.9 * n):]

# 3. Save splits
os.makedirs("../splits", exist_ok=True)

for name, ids in zip(
    ["train", "val", "test"],
    [train_ids, val_ids, test_ids]
):
    with open(f"../splits/{name}.txt", "w") as f:
        for i in ids:
            f.write(i + "\n")

print("Splits saved.")
print("Splits saved at:", os.path.abspath("../splits"))
