import pandas as pd
import os

# Create directories if they don't exist
os.makedirs('WiC/dev', exist_ok=True)
os.makedirs('WiC/test', exist_ok=True)

# Process dev.jsonl
print("Processing dev.jsonl...")
df_dev = pd.read_json('WiC/basqueWiC/dev.jsonl', lines=True)
# Convert labels from true/false to T/F
df_dev['label'] = df_dev['label'].map({True: 'T', False: 'F', 'true': 'T', 'false': 'F'})
# Extract the gold labels (label column) and save to gold file
gold_labels = df_dev['label'].astype(str)
gold_labels.to_csv('WiC/dev/dev.eu.gold.txt', index=False, header=False)

# Transform data to match Spanish WiC format: word, POS, indices, sentence1, sentence2
# Create indices column in format "start1-start2"
df_dev['indices'] = df_dev['start1'].astype(str) + '-' + df_dev['start2'].astype(str)
# Set POS to empty string (not provided in the data)
df_dev['pos'] = 'N'
# Reorder columns to match the format: word, pos, indices, sentence1, sentence2
data = df_dev[['word', 'pos', 'indices', 'sentence1', 'sentence2']]
data.to_csv('WiC/dev/dev.eu.data.txt', sep='\t', index=False, header=False)

print(f"Dev set processed: {len(df_dev)} instances")

# Process test.jsonl
print("Processing test.jsonl...")
df_test = pd.read_json('WiC/basqueWiC/test.jsonl', lines=True)


# Convert labels from true/false to T/F
df_test['label'] = df_test['label'].map({True: 'T', False: 'F', 'true': 'T', 'false': 'F'})

# Extract the gold labels (label column) and save to gold file
gold_labels = df_test['label'].astype(str)
gold_labels.to_csv('WiC/test/test.eu.gold.txt', index=False, header=False)

# Transform data to match Spanish WiC format: word, POS, indices, sentence1, sentence2
# Create indices column in format "start1-start2"
df_test['indices'] = df_test['start1'].astype(str) + '-' + df_test['start2'].astype(str)
# Set POS to empty string (not provided in the data)
df_test['pos'] = 'N'
# Reorder columns to match the format: word, pos, indices, sentence1, sentence2
data = df_test[['word', 'pos', 'indices', 'sentence1', 'sentence2']]
data.to_csv('WiC/test/test.eu.data.txt', sep='\t', index=False, header=False)

print(f"Test set processed: {len(df_test)} instances")
print("\nFiles created successfully!")
print("Output files:")
print("  - WiC/dev/dev.eu.gold.txt")
print("  - WiC/dev/dev.eu.data.txt")
print("  - WiC/test/test.eu.gold.txt")
print("  - WiC/test/test.eu.data.txt")
