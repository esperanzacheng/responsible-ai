import json
import urllib.request
import os

url = "https://huggingface.co/datasets/ZenMoore/RoleBench/resolve/main/rolebench-eng/instruction-generalization/role_specific/train.jsonl"
urllib.request.urlretrieve(url, "data/train.jsonl")

data = []
with open("data/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"\nLoaded {len(data)} samples")
print(f"Keys in first sample: {list(data[0].keys())}")

# Display first 3 samples
print("\nFirst 3 samples:")
for i, item in enumerate(data[:3], 1):
    print(f"\nSample {i}:")
    print(f"  Role: {item.get('role', 'N/A')}")
    print(f"  Question: {str(item.get('question', 'N/A'))[:100]}...")
          