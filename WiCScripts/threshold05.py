import json

def threshold05(input_file, output_file):
    with open(input_file, 'r') as f:
        all_data = f.readlines()
        data = [json.loads(line) for line in all_data]
        acc = [1.0 if (line["dot"] > 0.5 and line["tag"] == "T") or (line["dot"] <= 0.5 and line["tag"] == "F") else 0.0 for line in data]
        accuracy = sum(acc) / len(acc)
    with open(output_file, 'w') as f:
        f.write("Definition: " + str(accuracy) + "\n")

if __name__ == "__main__":
    model_names=["Llama2", "Llama3_8B", "Llama3_70B", "Gemma3_4B", "Gemma3_12B", "Gemma3_27B", "Qwen3_8B", "Qwen3_14B", "Qwen3_32B", "Latxa_70B", "Latxa_8B", "Qwen3_4B"]  # Update with your model names
    number_of_shots = [0,5]
    for modelname in model_names:
        for shots in number_of_shots:
            if shots == 5 and modelname == "Llama3LORA_DEF":
                continue
            threshold05(f'WiCOutputs/{shots}Shot/{modelname}_test.json', f'WiCOutputs/{shots}Shot/{modelname}_thr05.txt')
