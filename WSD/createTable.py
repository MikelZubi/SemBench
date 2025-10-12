import os
import pandas as pd
from collections import defaultdict

# Define the directory path

# Initialize an empty list to store the table data
table_data = []
#shots = [0,1,2,3,4,5]
shots = [0, 5]
all_tables = {"easy": defaultdict(lambda: []), "medium": defaultdict(lambda: []), "hard": defaultdict(lambda: []), "random": defaultdict(lambda: [])}
# Iterate over all subdirectories and files in the directory
for shot in shots:
    directory = 'WSDEOutputs/'+str(shot)+'Shot/'
    for file in os.listdir(directory):
        # Check if the file is a text file
        if file.endswith('.txt'):
            # Read the contents of the text file
            filename = file.split(".")[0]
            with open(os.path.join(directory, file), 'r') as f:
                text = f.readlines()
            
            # Append the rows to the table data
            for row in text:
                rows = row.split(": ")
                dificultie = rows[0]
                score = round(float(rows[1][:-1]) * 100,2)
                all_tables[dificultie][filename].append(score)

# Create a DataFrame from the table data
for key in all_tables:
    df = pd.DataFrame(columns=shots)
    for k, v in all_tables[key].items():
        if k == "OpenChat" or k== "Llama3LORA_DEF":
            continue
        if len(v) != len(shots):
            for i in range(len(shots)-len(v)):
                v.append(v[0])
        df.loc[k] = v
    df = df.transpose()
    df.index.name = 'Shot'
    #Export the DataFrame to a CSV file
    df.to_csv('WSDEOutputs/'+key+'.csv')


all_tables = {"easy": defaultdict(lambda: []), "medium": defaultdict(lambda: []), "hard": defaultdict(lambda: []), "random": defaultdict(lambda: [])}
# Iterate over all subdirectories and files in the directory
for shot in shots:
    directory = 'WSDOutputs/'+str(shot)+'Shot/'
    for file in os.listdir(directory):
        # Check if the file is a text file
        if file.endswith('.txt'):
            # Read the contents of the text file
            filename = file.split(".")[0]
            with open(os.path.join(directory, file), 'r') as f:
                text = f.readlines()
            
            # Append the rows to the table data
            for row in text:
                rows = row.split(": ")
                dificultie = rows[0]
                score = round(float(rows[1][:-1]) * 100,2)
                all_tables[dificultie][filename].append(score)

# Create a DataFrame from the table data
for key in all_tables:
    df = pd.DataFrame(columns=shots)
    for k, v in all_tables[key].items():
        if k == "OpenChat" or k== "Llama3LORA_DEF":
            continue
        if len(v) != len(shots):
            for i in range(len(shots)-len(v)):
                v.append(v[0])
        df.loc[k] = v
    df = df.transpose()
    df.index.name = 'Shot'
    #Export the DataFrame to a CSV file
    df.to_csv('WSDOutputs/'+key+'.csv')
