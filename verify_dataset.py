"""Quick script to verify dataset contents."""

from datasets import load_from_disk

# Load and inspect Atarashi dataset
atarashi = load_from_disk("output/atarashi")
print("=== ATARASHI DATASET ===")
print(f"Train: {len(atarashi['train']):,} samples")
print(f"Test:  {len(atarashi['test']):,} samples")
print(f"Columns: {atarashi['train'].column_names}")
print()
print("Sample row (train[0]):")
row = atarashi["train"][0]
print(f"  license_key: {row['license_key']}")
print(f"  text: {row['text'][:120]}...")
print(f"  source: {row['source']}")
print()

# Load and inspect Nirjas dataset
nirjas = load_from_disk("output/nirjas")
print("=== NIRJAS DATASET ===")
print(f"Train: {len(nirjas['train']):,} samples")
print(f"Test:  {len(nirjas['test']):,} samples")
print(f"Columns: {nirjas['train'].column_names}")
print()

# Show one license-related and one not-license-related
# Labels are integers: 1 = license_related, 0 = not_license_related
lr_found = False
nlr_found = False
for row in nirjas["train"]:
    if row["label"] == 1 and not lr_found:
        print("Sample license_related (label=1):")
        print(f"  text: {row['text'][:150]}...")
        print(f"  source: {row['source']}")
        print()
        lr_found = True
    if row["label"] == 0 and not nlr_found:
        print("Sample not_license_related (label=0):")
        print(f"  text: {row['text'][:150]}...")
        print(f"  source: {row['source']}")
        print(f"  negative_type: {row.get('negative_type', 'N/A')}")
        print()
        nlr_found = True
    if lr_found and nlr_found:
        break

# Label distribution verification
train_labels = nirjas["train"]["label"]
print("Label distribution (train):")
print(f"  license_related (1):     {sum(1 for lbl in train_labels if lbl == 1):,}")
print(f"  not_license_related (0): {sum(1 for lbl in train_labels if lbl == 0):,}")
