import os
import csv
import sys
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Create a summary table from model results.')
parser.add_argument('--language', type=str, help='The language of the evaluation: "EN" for English, "ES" for Spanish')
parser.set_defaults(language="EN")
language = parser.parse_args().language

if language == "EN":
    langcode = ""
    shots = [0,5]
else:
    langcode = language
    shots = [0,5]


models = ["Llama2", "Llama3_8B", "Llama3_70B", "Gemma3_4B", "Gemma3_12B", "Gemma3_27B", "Qwen3_4B","Qwen3_8B", "Qwen3_14B", "Qwen3_32B", "Latxa_8B", "Latxa_70B"]
# Create a list to store the results
results = []

# Iterate over the shots
for shot in shots:
    # Create a dictionary to store the results for each shot
    shot_results = {"Shot": shot}
    
    # Find the folders in WiCOutputs that start with the shot number
    folder_path = f"WiCOutputs{langcode}/{shot}Shot"
    
    # Iterate over the models
    for model in models:
        # Find the file that has the model name and "_results.txt" in the folder
        file_name = f"{model}_result.txt"
        file_path = None
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            # Read the file and save the contents
            if file_path:
                with open(file_path, "r") as file:
                    contents = file.readlines()
                    acc = contents[0][12:-1]
                    shot_results[model] = float(acc) * 100
            else:
                shot_results[model] = "N/A"
    
    # Append the shot results to the overall results list
    results.append(shot_results)
    print(results)

# Save the results to a CSV file
csv_file = f"WiCOutputs{langcode}/definition.csv"
fieldnames = ["Shot"] + models
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

results.clear()


# Iterate over the shots
for shot in shots:
    # Create a dictionary to store the results for each shot
    shot_results = {"Shot": shot}
    
    # Find the folders in WiCOutputs that start with the shot number
    folder_path = f"WiCOutputs{langcode}/{shot}Shot"
    
    # Iterate over the models
    for model in models:
        # Find the file that has the model name and "_results.txt" in the folder
        file_name = f"{model}_thr05.txt"
        file_path = None
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            # Read the file and save the contents
            if file_path:
                with open(file_path, "r") as file:
                    contents = file.readlines()
                    acc = contents[0][12:-1]
                    shot_results[model] = float(acc) * 100
            else:
                shot_results[model] = "N/A"
    
    # Append the shot results to the overall results list
    results.append(shot_results)
    print(results)

# Save the results to a CSV file
csv_file = f"WiCOutputs{langcode}/definition_thr05.csv"
fieldnames = ["Shot"] + models
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

results.clear()


# Iterate over the shots
for shot in shots:
    # Create a dictionary to store the results for each shot
    shot_results = {"Shot": shot}
    
    # Find the folders in WiCOutputs that start with the shot number
    folder_path = f"WiCOutputs/{shot}Shot"
    
    # Iterate over the models
    for model in models:
        # Find the file that has the model name and "_results.txt" in the folder
        file_name = f"{model}_result.txt"
        file_path = None
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            # Read the file and save the contents
            if file_path:
                with open(file_path, "r") as file:
                    contents = file.readlines()
                    acc = contents[1][22:-1]
                    shot_results[model] = float(acc) * 100
            else:
                shot_results[model] = "N/A"
    
    # Append the shot results to the overall results list
    results.append(shot_results)
    print(results)

# Save the results to a CSV file
csv_file = f"WiCOutputs{langcode}/definition+Context.csv"
fieldnames = ["Shot"] + models
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)