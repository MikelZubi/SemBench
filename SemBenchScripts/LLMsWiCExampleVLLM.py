from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
sys.path.append("prompts")
from prompt_factory import get_promptFactory
import os
import numpy as np
import argparse

from vllm import LLM, SamplingParams, inputs





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
    terminators = [tokenizer.eos_token_id]
    sampling_params = SamplingParams(temperature=0.0, stop_token_ids=terminators, max_tokens=2000, n=1)  
    example_prompts = []
    correct_defs = [def1s[i] if labels[i] == 0 else def2s[i] for i in range(len(words))]
    print("Generating example prompts")
    for i in tqdm(range(len(words))):
        example_prompts.append(inputs.TokensPrompt({"prompt_token_ids":PROMPT.generate_promptExampleDef(modelname,tokenizer,words[i],correct_defs[i], POSs[i], fewN, fewV, tokenize=True)}))
    
    print("Generating examples")
    results1 = llm.generate(example_prompts, sampling_params=sampling_params)
    generated_examples = []
    for output in results1:
        text = output.outputs[0].text
        if "</think>" in text:
            text = text.split("</think>")[-1]
        generated_examples.append(text)
    print("Generating definition prompts")
    definition_prompts = []
    for i in tqdm(range(len(words))):
        definition_prompts.append(inputs.TokensPrompt({"prompt_token_ids":PROMPT.generate_promptV2(modelname,tokenizer,words[i],generated_examples[i], POSs[i], fewN, fewV, tokenize=True)}))
    print("Generating definitions")
    results2 = llm.generate(definition_prompts, sampling_params=sampling_params)
    generated_definitions = []
    for output in results2:
        text = output.outputs[0].text
        if "</think>" in text:
            text = text.split("</think>")[-1]
        generated_definitions.append(text)
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
    parser = argparse.ArgumentParser(description='WSD evaluation script')
    parser.add_argument('--modelname', type=str, help='Name of the model to use')
    parser.add_argument('--k', type=int, help='Number of shots for few-shot learning')
    parser.add_argument('--language', type=str, help='The languages that will be evaluated: "EN" for English, "ES" for Spanish')

    parser.set_defaults(language="EN")
    parser.set_defaults(k=5)
    args = parser.parse_args()
    PROMPT = get_promptFactory(args.language)

    modelname = args.modelname
    k = args.k
    language = args.language
    if language == "ES":
        codepath = "es_"
        fewpath = language
    elif language == "EU":
        codepath = "eu_"
        fewpath = language
    else:
        codepath = ""
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
    sb = SentenceTransformer("google/embeddinggemma-300m")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = LLM(model=modelpath, tensor_parallel_size=4, gpu_memory_utilization=0.8)
    #pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id # Hack to fix a bug in transformers (batch_size)
    if language == "EN":
        dificulties = ["easy", "medium", "hard", "random"]
        words = {"easy":[], "medium":[], "hard":[], "random":[]}
        examples = {"easy":[], "medium":[], "hard":[], "random":[]}
        POSs = {"easy":[], "medium":[], "hard":[], "random":[]}
        labels = {"easy":[], "medium":[], "hard":[], "random":[]}
        def1s = {"easy":[], "medium":[], "hard":[], "random":[]}
        def2s = {"easy":[], "medium":[], "hard":[], "random":[]}
    else: 
        dificulties = ["random"]
        words = {"random":[]}
        examples = {"random":[]}
        POSs = {"random":[]}
        labels = {"random":[]}
        def1s = {"random":[]}
        def2s = {"random":[]}
    filenameResult = "WSDEOutputs"+fewpath+"/" + str(k) + "Shot/" + modelname + ".txt"
    os.makedirs(os.path.dirname(filenameResult), exist_ok=True) 
    dotvalues = {}
    for dificultie in dificulties:
        with open("WSD/data/test_"+codepath+dificultie+".json","r") as Wsd_data:
            Wsd = Wsd_data.read().splitlines()
            for i in range(len(Wsd)):
                data = json.loads(Wsd[i])
                words[dificultie].append(data["word"])
                examples[dificultie].append(data["example"])
                POSs[dificultie].append(data["POS"])
                labels[dificultie].append(data["label"])
                def1s[dificultie].append(data["definitions"][0])
                def2s[dificultie].append(data["definitions"][1])
        
        filename = "WSDEOutputs"+fewpath+"/" + str(k) + "Shot/" + modelname +"_"+dificultie+".json"
        
        useModels(llm,tokenizer,sb,filename,words[dificultie], examples[dificultie], POSs[dificultie], labels[dificultie], def1s[dificultie], def2s[dificultie], modelname, fewN, fewV)
        dotvalues[dificultie] = estimate(filename)
        
    file = open(filenameResult, "w",encoding='utf-8')
    for dificulti in dificulties:
        file.write(dificulti + ": " + str(dotvalues[dificulti]) + "\n")
    
    
