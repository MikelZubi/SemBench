from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from generatePrompt import generate_promptV2, generate_promptExampleDef
import os
import numpy as np

from vllm import LLM, SamplingParams






'''
def testModels(word, example, POS, label, def1,def2, pipeline, tokenizer, sb, file, modelname, fewN, fewV):

    def_correct = def1 if label == 0 else def2
    promptExample = generate_promptExampleDef(modelname,tokenizer,word,def_correct, POS, fewN, fewV)
    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    sequences = pipeline(
        [promptExample],
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=140,
        batch_size = 1, #Put 1 if error
    )
    seq0 = sequences[0]
    output0 = seq0[0]['generated_text']
    exampleG = output0[len(promptExample):]
    prompt = generate_promptV2(modelname,tokenizer,word,exampleG, POS, fewN, fewV)
    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    sequences = pipeline(
        [prompt],
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=140,
        batch_size = 1, #Put 1 if error
    )

    seq1 = sequences[0]
    output1 = seq1[0]['generated_text']
    defG = output1[len(prompt):]
    print(defG)

    embeddingG = sb.encode(defG, convert_to_tensor=True)
    embedding1 = sb.encode(def1, convert_to_tensor=True)
    embedding2 = sb.encode(def2, convert_to_tensor=True)


    dot_score1 = util.dot_score(embeddingG, embedding1)[0][0].item()
    dot_score2 = util.dot_score(embeddingG, embedding2)[0][0].item()
    if dot_score1 > dot_score2:
        pred_label = 0
        dot_score = dot_score1
    else:
        pred_label = 1
        dot_score = dot_score2
    dictionary = {"word": word, "POS": POS, "example": example, "defG": defG, "def1": def1 , "def2": def2, "dot": dot_score, "label": label, "pred_label": pred_label}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")
'''

def useModels(llm,tokenizer,sb,filename,words, examples, POSs, labels, def1s,def2s,modelname,fewN, fewV):
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(temperature=0.0, stop_token_ids=terminators, max_tokens=140, n=1)  
    example_prompts = []
    correct_defs = [def1s[i] if labels[i] == 0 else def2s[i] for i in range(len(words))]
    print("Generating example prompts")
    for i in tqdm(range(len(words))):
        example_prompts.append(generate_promptExampleDef(modelname,tokenizer,words[i],correct_defs[i], POSs[i], fewN, fewV, tokenize=True))
    

    
    print("Generating examples")
    results1 = llm.generate(prompt_token_ids=example_prompts, sampling_params=sampling_params)
    generated_examples = []
    for output in results1:
        generated_examples.append(output.outputs[0].text)
    print("Generating definition prompts")
    definition_prompts = []
    for i in tqdm(range(len(words))):
        definition_prompts.append(generate_promptV2(modelname,tokenizer,words[i],generated_examples[i], POSs[i], fewN, fewV, tokenize=True))
    print("Generating definitions")
    results2 = llm.generate(prompt_token_ids=definition_prompts, sampling_params=sampling_params)
    generated_definitions = []
    for output in results2:
        generated_definitions.append(output.outputs[0].text)
    print("Scoring")
    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        defG = generated_definitions[i]
        def1 = def1s[i]
        def2 = def2s[i]
        label = labels[i]
        word = words[i]
        example = examples[i]
        PoS = POSs[i]
        examplesG = generated_examples[i]
        embeddingG = sb.encode(defG, convert_to_tensor=True)
        embedding1 = sb.encode(def1, convert_to_tensor=True)
        embedding2 = sb.encode(def2, convert_to_tensor=True)


        dot_score1 = util.dot_score(embeddingG, embedding1)[0][0].item()
        dot_score2 = util.dot_score(embeddingG, embedding2)[0][0].item()
        if dot_score1 > dot_score2:
            pred_label = 0
            dot_score = dot_score1
        else:
            pred_label = 1
            dot_score = dot_score2
        dictionary = {"word": word, "POS": PoS, "example": example, "examplesG": examplesG, "defG": defG, "def1": def1 , "def2": def2, "dot": dot_score, "label": label, "pred_label": pred_label}
        file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")
    file.close()


def estimate(path):
    count = 0.0
    with open(path, "r") as file:
        file_data = file.readlines()
        length = len(file_data)
        for line in file_data:
            data = json.loads(line)
            if data["label"] == data["pred_label"]:
                count += 1
                
    return count/length



if __name__ == "__main__":
    rd.seed(16)
    modelname = sys.argv[1]
    k = int(sys.argv[2])
    if sys.argv[3] == "WN":
        fewpath = "WN"
    else:
        fewpath = ""
    with open("modelsData.json", "r") as jsonfile:
        modelsdata = json.load(jsonfile)
        modelpath = modelsdata[modelname]["path"]

    fewN = {}
    fewN["k"] = k
    fewN["words"] = []
    fewN["definitions"] = []
    fewN["examples"] = []
    nouns = open("polysemicNouns"+fewpath+".json","r").read().splitlines()
    for i in range(k):
        line = rd.choice(nouns)
        dataR = json.loads(line)
        examples = dataR["examples"]
        pos = dataR["POS"]
        word = dataR["word"]
        definizioa = dataR["definition"]
        fewN["words"].append(word)
        fewN["definitions"].append(definizioa)
        fewN["examples"].append(examples)
        nouns.remove(line)
    fewV = {}
    fewV["k"] = k
    fewV["words"] = []
    fewV["definitions"] = []
    fewV["examples"] = []
    verbs = open("polysemicVerbs"+fewpath+".json","r").read().splitlines()
    for i in range(k):
        line = rd.choice(verbs)
        dataR = json.loads(line)
        examples = dataR["examples"]
        pos = dataR["POS"]
        word = dataR["word"]
        definizioa = dataR["definition"]
        fewV["words"].append(word)
        fewV["definitions"].append(definizioa)
        fewV["examples"].append(examples)
        verbs.remove(line)

    tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side='right')
    sb = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if modelname == "Llama3_70B":
        llm = LLM(model=modelpath, tensor_parallel_size=2, gpu_memory_utilization=0.9)
    else:
        llm = LLM(model=modelpath, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    #pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id # Hack to fix a bug in transformers (batch_size)
    dificulties = ["easy", "medium", "hard", "random"]
    words = {"easy":[], "medium":[], "hard":[], "random":[]}
    examples = {"easy":[], "medium":[], "hard":[], "random":[]}
    POSs = {"easy":[], "medium":[], "hard":[], "random":[]}
    labels = {"easy":[], "medium":[], "hard":[], "random":[]}
    def1s = {"easy":[], "medium":[], "hard":[], "random":[]}
    def2s = {"easy":[], "medium":[], "hard":[], "random":[]}
    filenameResult = "WSDEOutputs/" + str(k) + "Shot/" + modelname + ".txt"
    os.makedirs(os.path.dirname(filenameResult), exist_ok=True) 
    dotvalues = {}
    for dificultie in dificulties:
        with open("WSD/data/test_"+dificultie+".json","r") as Wsd_data:
            Wsd = Wsd_data.read().splitlines()
            for i in range(len(Wsd)):
                data = json.loads(Wsd[i])
                words[dificultie].append(data["word"])
                examples[dificultie].append(data["example"])
                POSs[dificultie].append(data["POS"])
                labels[dificultie].append(data["label"])
                def1s[dificultie].append(data["definitions"][0])
                def2s[dificultie].append(data["definitions"][1])
        
        filename = "WSDEOutputs/" + str(k) + "Shot/" + modelname +"_"+dificultie+".json"
        
        useModels(llm,tokenizer,sb,filename,words[dificultie], examples[dificultie], POSs[dificultie], labels[dificultie], def1s[dificultie], def2s[dificultie], modelname, fewN, fewV)
        dotvalues[dificultie] = estimate(filename)
        
    file = open(filenameResult, "w",encoding='utf-8')
    for dificulti in dificulties:
        file.write(dificulti + ": " + str(dotvalues[dificulti]) + "\n")
    
    
