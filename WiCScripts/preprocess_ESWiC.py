import pandas as pd
import os

# Create directories if they don't exist
os.makedirs('WiC/dev', exist_ok=True)

# Read the TSV file
df = pd.read_csv('WiC/wic-pln-2025/spa_dev.tsv', sep='\t')

# Extract the gold labels (last column) and save to gold file
gold_labels = df.iloc[:, -1]
gold_labels.to_csv('WiC/dev/dev.es.gold.txt', index=False, header=False)

# Extract all columns except the last one and save to data file
data = df.iloc[:, :-1]
data.to_csv('WiC/dev/dev.es.data.txt', sep='\t', index=False)


# Read the TSV file
df = pd.read_csv('WiC/wic-pln-2025/spa_test.tsv', sep='\t')

# Extract the gold labels (last column) and save to gold file
gold_labels = df.iloc[:, -1]
gold_labels.to_csv('WiC/test/test.es.gold.txt', index=False, header=False)

# Extract all columns except the last one and save to data file
data = df.iloc[:, :-1]
data.to_csv('WiC/test/test.es.data.txt', sep='\t', index=False)

print("Files created successfully!")