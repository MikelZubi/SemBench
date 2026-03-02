from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from prompts.prompt_factory import get_promptFactory
import os
import numpy as np
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import argparse





def testModels(word, example, POS, label, def1,def2, pipeline, tokenizer, sb, file, modelname, fewN, fewV):

    if modelname == "Llama3_70B" or modelname == "Llama3LORA_DEF":
        batch_size = 1
    else:
        batch_size = 2
    prompt1 = PROMPT.generate_promptV2(modelname,tokenizer,word,example, POS, fewN, fewV)
    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    sequences = pipeline(
        [prompt1],
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=140,
        batch_size = batch_size,
    )

    seq1 = sequences[0]

    output1 = seq1[0]['generated_text']

    defG = output1[len(prompt1):]

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


def useModels(pipeline,tokenizer,sb,filename,words, examples, POSs, labels, def1s,def2s,modelname,fewN, fewV):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], examples[i],POSs[i], labels[i], def1s[i], def2s[i],pipeline, tokenizer, sb, file, modelname, fewN, fewV)
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

    modelname = args.model
    k = args.k
    language = args.language
    if language == "ES":
        codepath = "es_"
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
    if modelname == "Llama3_70B":
        #quantization_config = QuantoConfig(weights="int8")
        model = AutoModelForCausalLM.from_pretrained(modelpath,load_in_8bit=True)
        pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype = torch.bfloat16)
        pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        device=0
        )
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
        dificutlies = ["random"]
        words = {"random":[]}
        examples = {"random":[]}
        POSs = {"random":[]}
        labels = {"random":[]}
        def1s = {"random":[]}
        def2s = {"random":[]}
    filenameResult = "WSDOutputs"+fewpath+"/" + str(k) + "Shot/" + modelname + ".txt"
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
        
        filename = "WSDOutputs"+language+"/" + str(k) + "Shot/" + modelname +"_"+dificultie+".json"
        
        useModels(pipeline,tokenizer,sb,filename,words[dificultie], examples[dificultie], POSs[dificultie], labels[dificultie], def1s[dificultie], def2s[dificultie], modelname, fewN, fewV)
        dotvalues[dificultie] = estimate(filename)
        
    file = open(filenameResult, "w",encoding='utf-8')
    for dificulti in dificulties:
        file.write(dificulti + ": " + str(dotvalues[dificulti]) + "\n")
    file.close()
    
    
