import os
import pandas as pd
from collections import defaultdict
import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(description='Process WSD outputs and create CSV tables')
parser.add_argument('--language', type=str, help='Language to process (default: english)')
parser.set_defaults(language='EN')
args = parser.parse_args()

language = args.language

# Define difficulties based on language
if language == 'EN':
    code_lang = ""
    difficulties = ["easy", "medium", "hard", "random"]
    all_tables = {"easy": defaultdict(lambda: []), "medium": defaultdict(lambda: []), "hard": defaultdict(lambda: []), "random": defaultdict(lambda: [])}
    shots = [0,5]

else:
    code_lang = language
    difficulties = ["random"]
    all_tables = {"random": defaultdict(lambda: [])}
    shots = [0,5]
# Define the directory path

# Initialize an empty list to store the table data
table_data = []
#shots = [0,1,2,3,4,5]

# Iterate over all subdirectories and files in the directory
for shot in shots:
    directory = f'WSDEOutputs{code_lang}/{shot}Shot/'
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
                score = float(rows[1][:-1]) * 100
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
    df.to_csv(f'WSDEOutputs{code_lang}/'+key+'.csv')


if language == 'EN':
    difficulties = ["easy", "medium", "hard", "random"]
    all_tables = {"easy": defaultdict(lambda: []), "medium": defaultdict(lambda: []), "hard": defaultdict(lambda: []), "random": defaultdict(lambda: [])}

    # Iterate over all subdirectories and files in the directory
    for shot in shots:
        directory = f'WSDOutputs{code_lang}/{shot}Shot/'
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
                    score = float(rows[1][:-1]) * 100
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
        df.to_csv(f'WSDOutputs{code_lang}/'+key+'.csv')
