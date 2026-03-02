import json
import random

# Leer el archivo JSONL
input_file = 'dictionarys/eeh_preprocess.json'
output_file = 'WSD/data/test_eu_random.json'

# Diccionario para agrupar palabras polisémicas
polysemic_words = {}

# Leer y agrupar por palabra
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line.strip())
        word = entry['word']
        
        if word not in polysemic_words:
            polysemic_words[word] = []
        polysemic_words[word].append(entry)

# Crear dataset con palabras que tienen 2 o más definiciones
dataset = []

for word, entries in polysemic_words.items():
    if len(entries) >= 2:
        # Si hay más de 2 definiciones, seleccionar 2 aleatorias
        if len(entries) > 2:
            selected_entries = random.sample(entries, 2)
        else:
            selected_entries = entries
        
        # Seleccionar aleatoriamente cuál será la definición correcta (0 o 1)
        label = random.randint(0, 1)
        
        dataset_entry = {
            "word": word,
            "POS": selected_entries[label].get("POS", ""),
            "definitions": [
                selected_entries[0]["definition"],
                selected_entries[1]["definition"]
            ],
            "example": selected_entries[label].get("examples", ""),
            "label": label
        }
        
        dataset.append(dataset_entry)

# Guardar el dataset
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Dataset creado con {len(dataset)} entradas polisémicas")
print(f"Guardado en: {output_file}")