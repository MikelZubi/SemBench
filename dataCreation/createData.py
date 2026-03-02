import json
import random as rd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
import argparse


#Select one random definition and example from data that it is between j and max
def select_def_exp(data,j,max, polysem = False, exclude = [-1]):
    mod = max - j
    lag = rd.randrange(0,mod)
    if polysem:
        length = 0
    else:
        length = 1
    for i in range(0,mod):
        idx = ((lag + i) % mod) + j

        if idx in exclude:
            continue
        if len(data[idx]['examples']) > length:
            return idx
    return -1

def sort_st(def_correct, model, definition):
    embd1 = model.encode(def_correct, convert_to_tensor=True)
    embd2 = model.encode(definition, convert_to_tensor=True)
    dot_score = util.dot_score(embd1, embd2)[0][0].item()
    return dot_score


def select_def(definitions, dificulty):
    if dificulty == "easy":
        return definitions[0]
    elif dificulty == "medium":
        return definitions[int(len(definitions)/2)]
    elif dificulty == "hard":
        return definitions[-1]
    elif dificulty == "random":
        return rd.choice(definitions)

def select_def_na(definitions, dificulty):
    if dificulty == "easy":
        return definitions[0], definitions[1]
    elif dificulty == "medium":
        return definitions[int(len(definitions)/2)], definitions[int(len(definitions)/2)-1]
    elif dificulty == "hard":
        return definitions[-1], definitions[-2]
    elif dificulty == "random":
        rd1 = rd.choice(definitions)
        definitions.remove(rd1)
        rd2 = rd.choice(definitions)
        return rd1, rd2

def sampler():
    return rd.choice([0,1,3]) == 0

def get_polysemic_na(data,position,model,dificulty):
    j = position
    word = data[j]['word']
    pos = data[j]['POS']
    while data[position]['POS'] == data[j]['POS'] and data[position]['word'] == data[j]['word'] and "Definition not found $REF:" not in data[j]["definition"] and j > 0:
        j -= 1
    j += 1
    max = j
    while data[position]['POS'] == data[max]['POS'] and data[position]['word'] == data[max]['word'] and "Definition not found $REF:" not in data[max]["definition"] and (max + 1) < len(data):
        max += 1
    if max - j < 3:
        return {}, {}
    idx = select_def_exp(data,j,max,polysem=False)
    idx2 = select_def_exp(data,j,max,polysem=True,exclude=[idx]) #OWiC 
    if idx == -1 or idx2 == -1:
        return {}, {}
    definitions = []
    for i in range(j,max):
        if i == idx:
            continue
        definitions.append(data[i]['definition'])
    definitions.sort(key=lambda x: sort_st(data[idx]['definition'], model, x))
    definition1, definition2 = select_def_na(definitions, dificulty)
    example = rd.choice(data[idx]['examples'])
    def_correct = data[idx]['definition']
    if sampler():
        defs = [definition2, definition1]
        rd.shuffle(defs)
        label = -1
    else: 
        defs = [definition1, def_correct]
        rd.shuffle(defs)
        label = defs.index(def_correct)
    word = data[idx]['word']
    pos = data[idx]['POS']
    resultWSD = {"word":word, 'POS':pos, "label": label, "definitions": defs, "example": example}
    examples_same = rd.sample(data[idx]['examples'], 2)
    example1 = examples_same[0]
    example2 = examples_same[1]
    example3 = rd.choice(data[idx2]['examples'])  
    examples = [example1, example2, example3]
    rd.shuffle(examples)
    resultOWiC = {"word":word, 'POS':pos, "sentence0": examples[0], "sentence1": examples[1], "sentence2": examples[2], "label": examples.index(example3)}
    return resultWSD, resultOWiC

   


    

def get_polysemic(data,position,model,dificulty,polysem):
    j = position
    word = data[j]['word']
    pos = data[j]['POS']
    while data[position]['POS'] == data[j]['POS'] and data[position]['word'] == data[j]['word'] and j > 0:
        j -= 1
    j += 1
    max = j
    while data[position]['POS'] == data[max]['POS'] and data[position]['word'] == data[max]['word'] and (max + 1) < len(data):
        max += 1
    idx = select_def_exp(data,j,max,polysem=polysem)
    if idx == -1:
        return {}, {}, {}
    if polysem:
        idx2 = select_def_exp(data,j,max,polysem=polysem,exclude=[idx])
        if idx2 == -1:
            return {}, {}, {}

    def_correct = data[idx]['definition']
    example = rd.choice(data[idx]['examples'])
    definitions = []
    for i in range(j,max):
        if i == idx:
            continue
        if "Definition not found $REF:" not in data[i]["definition"]:
            definitions.append(data[i]['definition'])
    if len(definitions) < 1:
        return {}, {}, {}
    
    definitions.sort(key=lambda x: sort_st(def_correct, model, x))
    definition = select_def(definitions, dificulty)
    defs = [definition, def_correct]
    rd.shuffle(defs)
    label = defs.index(def_correct)
    resultWSD = {"word":word, 'POS':pos, "label": label, "definitions": defs, "example": example}
    if polysem:
        example2 = rd.choice(data[idx2]['examples'])
        def_correct2 = data[idx2]['definition']
        resultWiC = {"word":word, 'POS':pos, "sentence1": example, "sentence2": example2, "definition1": def_correct, "definition2": def_correct2, "label": False}
    else:
        examples = rd.sample(data[idx]['examples'], 2)
        example1 = examples[0]
        example2 = examples[1]
        resultWiC = {"word":word, 'POS':pos, "sentence1": example1, "sentence2": example2, "definition1": def_correct, "definition2": def_correct, "label": True}
    
    #Bench
    resultsBench = {"word":word, 'POS':pos,  "sentence": example, "definition": def_correct}
    return resultWSD, resultWiC, resultsBench



# Load the pre-trained model
model = SentenceTransformer('google/embeddinggemma-300m')
argparser = argparse.ArgumentParser()
argparser.add_argument('--language', type=str, help='Language code')
argparser.set_defaults(language='EN')
args = argparser.parse_args()

leng = 1200
rd.seed(42)

language = args.language
if language == "EN":
    corpus_path = 'CorpusOxford.json'
    code_path = ""
    few_words = ["run","strike","turn","fall","cut","hold","write","bank","party","date","head","ring","key","mouse"]
elif language == "ES":
    corpus_path = 'dictionarys/rae_preprocess.json'
    code_path = "es_"
    few_words = ["hoja", "banco", "boca", "ratón","sierra", "clavar", "pinchar", "pegar", "romper", "colgar"]
elif language == "EU":
    corpus_path = 'dictionarys/eeh_preprocess.json'
    code_path = "eu_"
    few_words = ["sagu", "giltz", "adar", "begi", "buru", "joan", "ekarri", "ikusi", "jan", "ahaztu"]


# Read the OxfordData.json file
with open(corpus_path) as file:
    data = [json.loads(line) for line in file.readlines()]

indexes = [i for i in range(len(data))]
difficulties = ["easy","medium","hard","random"]
all_data_WSD = {"easy":[],"medium":[],"hard":[],"random":[]}
all_data_WSD_na = {"easy":[],"medium":[],"hard":[],"random":[]}
all_data_WiC = []
all_data_WiC_na = []
all_data_Bench = []
appended = {"words":[],"POS":[],"line":[]}
counter = 0
poly = False
valid_POS = ["noun", "verb", "sustantivo", "verbo", "izena", "aditza"]


with tqdm(total=leng) as pbar:
    while len(all_data_WiC) < leng and indexes != []:
        i = rd.choice(indexes)
        indexes.remove(i)
        if data[i]['POS'] not in valid_POS or data[i]['word'] in appended["words"] or data[i]["word"] in few_words:
            continue
        for difficultie in difficulties:
            polyWSD_na, polyWiC_na = get_polysemic_na(data,i,model,difficultie)
            if polyWSD_na == {}:
                continue
            polyWSD, polyWiC, polyBench = get_polysemic(data,i,model,difficultie,poly)
            all_data_WSD[difficultie].append(polyWSD)
            all_data_WSD_na[difficultie].append(polyWSD_na)
            if difficultie == "easy":
                all_data_WiC.append(polyWiC)
                all_data_WiC_na.append(polyWiC_na)
                all_data_Bench.append(polyBench)
                appended["words"].append(data[i]['word'])
                appended["POS"].append(data[i]['POS'])
                appended["line"].append(i)
                pbar.update(1)
                counter += 1
                if counter >= 500:
                    poly = True


rd.shuffle(all_data_WiC)
rd.shuffle(all_data_WiC_na)
rd.shuffle(all_data_Bench)
rd.shuffle(all_data_WSD["easy"])
rd.shuffle(all_data_WSD["medium"])
rd.shuffle(all_data_WSD["hard"])
rd.shuffle(all_data_WSD["random"])
rd.shuffle(all_data_WSD_na["easy"])
rd.shuffle(all_data_WSD_na["medium"])
rd.shuffle(all_data_WSD_na["hard"])
rd.shuffle(all_data_WSD_na["random"])

wsd_path = "WSD/data/"
wsd_na_path = "WSD_na/data/"
OWiC_path = "OWiC/data/"
OWiC_na_path = "OWiC_na/data/"
Bench_path = "Bench/data/"
os.makedirs(wsd_path, exist_ok=True)
os.makedirs(wsd_na_path, exist_ok=True)
os.makedirs(OWiC_path, exist_ok=True)
os.makedirs(OWiC_na_path, exist_ok=True)
os.makedirs(Bench_path, exist_ok=True)

# Write polysemics to data.json file
for difficultie in difficulties:
    with open(wsd_path+"dev_"+code_path+difficultie+'.json', 'w') as outfile:
        for poly in all_data_WSD[difficultie][:200]:
            line = json.dumps(poly, ensure_ascii=False)
            outfile.write(line + '\n')
    with open(wsd_path+"test_"+code_path+difficultie+'.json', 'w') as outfile:
        for poly in all_data_WSD[difficultie][200:]:
            line = json.dumps(poly, ensure_ascii=False)
            outfile.write(line + '\n')
    with open(wsd_na_path+"dev_"+code_path+difficultie+'.json', 'w') as outfile:
        for poly in all_data_WSD_na[difficultie][:200]:
            line = json.dumps(poly, ensure_ascii=False)
            outfile.write(line + '\n')
    with open(wsd_na_path+"test_"+code_path+difficultie+'.json', 'w') as outfile:
        for poly in all_data_WSD_na[difficultie][200:]:
            line = json.dumps(poly, ensure_ascii=False)
            outfile.write(line + '\n')
with open(OWiC_path+'dev_'+code_path+'.json', 'w') as outfile:
    for poly in all_data_WiC[:200]:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')
with open(OWiC_path+'test_'+code_path+'.json', 'w') as outfile:
    for poly in all_data_WiC[200:]:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')
with open(OWiC_na_path+'dev_'+code_path+'.json', 'w') as outfile:
    for poly in all_data_WiC_na[:200]:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')
with open(OWiC_na_path+'test_'+code_path+'.json', 'w') as outfile:
    for poly in all_data_WiC_na[200:]:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')
with open(Bench_path+'dev_'+code_path+'.json', 'w') as outfile:
    for poly in all_data_Bench[:200]:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')
with open(Bench_path+'test_'+code_path+'.json', 'w') as outfile:
    for poly in all_data_Bench[200:]:
        line = json.dumps(poly, ensure_ascii=False)
        outfile.write(line + '\n')
with open('used_words.json', 'w') as outfile:
    for i in range(len(appended["words"])):
        data = {"word":appended["words"][i], "POS":appended["POS"][i], "line":appended["line"][i]}
        line = json.dumps(data, ensure_ascii=False)
        outfile.write(line + '\n')