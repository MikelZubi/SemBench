from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import transformers
import torch
import random as rd
import json 
from prompt_factory import get_promptFactory
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, accuracy_score
import os
import numpy as np
import argparse

from vllm import LLM, SamplingParams



def random_line(fname):
    lines = open(fname).read().splitlines()
    return rd.choice(lines)


def testModels(word, example1, example2, POS, tag, pipeline, tokenizer, sb, file, modelname, fewN, fewV):

    if modelname == "Llama3_70B" or modelname == "Llama3LORA_DEF":
        batch_size = 1
    else:
        batch_size = 2

    prompt1 = PROMPT.generate_promptV2(modelname,tokenizer,word,example1, POS, fewN, fewV)
    prompt2 = PROMPT.generate_promptV2(modelname,tokenizer,word,example2, POS, fewN, fewV)      
    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    sequences = pipeline(
        [prompt1,prompt2],
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=140,
        batch_size = batch_size, #Put 1 if error
    )

    seq1 = sequences[0]
    seq2 = sequences[1]
    output1 = seq1[0]['generated_text']
    output2 = seq2[0]['generated_text']
    def1 = output1[len(prompt1):]
    if "assistant\n\n" == def1[:11]:
        def1 = def1[11:]
    elif "assistant\n" == def1[:10]:
        def1 = def1[10:]
    def2 = output2[len(prompt2):]
    if "assistant\n\n" == def2[:11]:
        def2 = def2[11:]
    elif "assistant\n" == def2[:10]:
        def2 = def2[10:]
    embeddings1 = sb.encode(def1, convert_to_tensor=True)
    embeddings2 = sb.encode(def2, convert_to_tensor=True)
    defexample1 = "Word: " + word + "\nDefinition: " + def1 + "\nExample: " + example1
    defexample2 = "Word: " + word + "\nDefinition: " + def2 + "\nExample: " + example2
    embeddingsE1 = sb.encode(defexample1, convert_to_tensor=True)
    embeddingsE2 = sb.encode(defexample2, convert_to_tensor=True)
    dot_score = util.dot_score(embeddings1, embeddings2)[0][0].item()
    dot_scoreE = util.dot_score(embeddingsE1, embeddingsE2)[0][0].item()
    dictionary = {"word": word, "POS": POS, "sentence1": example1, "sentence2": example2, "def1": def1 , "def2": def2, "dot": dot_score, "dotE": dot_scoreE, "tag": tag}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(llm,tokenizer,sb,filename,words,sentences1,sentences2,POSs,tags,modelname,fewN, fewV):

    
    print("Generate prompts")
    prompt_sentence_1 = []
    prompt_sentence_2 = []
    for i in tqdm(range(len(words))):
        prompt_sentence_1.append(PROMPT.generate_promptV2(modelname,tokenizer,words[i],sentences1[i], POSs[i], fewN, fewV, tokenize=True))
        prompt_sentence_2.append(PROMPT.generate_promptV2(modelname,tokenizer,words[i],sentences2[i], POSs[i], fewN, fewV, tokenize=True))
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(temperature=0.0, stop_token_ids=terminators, max_tokens=140, n=1)
    print("Generating definitions sentence 1")
    results1 = llm.generate(prompt_token_ids=prompt_sentence_1, sampling_params=sampling_params)
    definitions1 = []
    for output in results1:
        definitions1.append(output.outputs[0].text)
    
    print("Generating definitions sentence 2")
    results2 = llm.generate(prompt_token_ids=prompt_sentence_2, sampling_params=sampling_params)
    definitions2 = []
    for output in results2:
        definitions2.append(output.outputs[0].text)
    
    print("Evaluating")
    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        def1 = definitions1[i]
        def2 = definitions2[i]
        embedding1 = sb.encode(def1, convert_to_tensor=True)
        embedding2 = sb.encode(def2, convert_to_tensor=True)
        defexample1 = "Word: " + words[i] + "\nDefinition: " + def1 + "\nExample: " + sentences1[i]
        defexample2 = "Word: " + words[i] + "\nDefinition: " + def2 + "\nExample: " + sentences2[i]
        embeddingE1 = sb.encode(defexample1, convert_to_tensor=True)
        embeddingE2 = sb.encode(defexample2, convert_to_tensor=True)
        dot_score = util.dot_score(embedding1, embedding2)[0][0].item()
        dot_scoreE = util.dot_score(embeddingE1, embeddingE2)[0][0].item()
        word = words[i]
        example1 = sentences1[i]
        example2 = sentences2[i]
        PoS = POSs[i]
        tag = tags[i]
        if dot_score > dot_scoreE:
            dot_scoreE = dot_score
        dictionary = {"word": word, "POS": PoS, "sentence1": example1, "sentence2": example2, "def1": def1 , "def2": def2, "dot": dot_score, "dotE": dot_scoreE, "tag": tag}
        file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")
    file.close()

def processWiC(line):
    lineDic = {}
    line = line.split("\t")
    lineDic["word"] = line[0]
    lineDic["POS"] = line[1]
    lineDic["sentence1"] = line[3]
    lineDic["sentence2"] = line[4]
    return lineDic 

def calculateThrshold(path, key):

    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append([data[key]])
                label_all.append(tvalue)
            else:
                all_score.append([data[key]])
                label_all.append(fvalue)
                
            

    _, _, thresholds  = precision_recall_curve(label_all, all_score)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(label_all, [m > thresh for m in all_score]))

    accuracies = np.array(accuracy_scores)
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    return max_accuracy_threshold

def estimate(path, thresh, key):
    file_path = path


    all_score = []
    label_all = []

    tvalue = True
    fvalue = False
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["tag"] == "T":
                all_score.append([data[key]])
                label_all.append(tvalue)
            else:
                all_score.append([data[key]])
                label_all.append(fvalue)
                
            
    return accuracy_score(label_all, [m > thresh for m in all_score])



if __name__ == "__main__":
    rd.seed(16)
    parser = argparse.ArgumentParser(description='WSD evaluation script')
    parser.add_argument('model', type=str, help='Name of the model to use')
    parser.add_argument('k', type=int, help='Number of shots for few-shot learning')
    parser.add_argument('language', type=str, help='The languages that will be evaluated: "EN" for English, "ES" for Spanish')

    parser.set_defaults(language="EN")
    parser.set_defaults(k=5)
    args = parser.parse_args()

    PROMPT = get_promptFactory(args.language)

    modelname = args.model
    k = args.k
    language = args.language
    if language == "ES":
        data_dev = "WiC/dev/dev.es.data.txt"
        data_test = "WiC/test/test.es.data.txt"
        gold_dev = "WiC/dev/dev.es.gold.txt"
        gold_test = "WiC/test/test.es.gold.txt"
        fewpath = "ES"
    else:
        data_dev = "WiC/dev/dev.data.txt"
        data_test = "WiC/test/test.data.txt"
        gold_dev = "WiC/dev/dev.gold.txt"
        gold_test = "WiC/test/test.gold.txt"
        fewpath = ""
    with open("modelsData.json", "r") as jsonfile:
        modelsdata = json.load(jsonfile)
        modelpath = modelsdata[modelname]["path"]
    filename = "WiCOutputs"+language+"/" + str(k) + "Shot/" + modelname + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    adiblen = 1
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
    words = []
    sentences1 = []
    sentences2 = []
    POSs = []
    tags = []
    WiC = open(data_dev,"r").read().splitlines()
    gold = open(gold_dev,"r").read().splitlines()
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])
    filenameDev = filename[:-5] + "_dev"+fewpath+".json"
    filenameTest = filename[:-5] + "_test"+fewpath+".json"
    filenameResult = filename[:-5] + "_result"+fewpath+".txt"

    tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side='right')
    sb = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if modelname == "Llama3_70B":
        llm = LLM(model=modelpath, tensor_parallel_size=2, gpu_memory_utilization=0.9)
    else:
        llm = LLM(model=modelpath, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    #model = model.to("cuda")
    #model.eval()
    #pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id # Hack to fix a bug in transformers (batch_size)
    useModels(llm,tokenizer,sb,filenameDev,words,sentences1,sentences2,POSs,tags,modelname,fewN, fewV)
    regrDot = calculateThrshold(filenameDev, "dot")
    regrDotE = calculateThrshold(filenameDev, "dotE")
    WiC = open(data_test,"r").read().splitlines()
    gold = open(gold_test,"r").read().splitlines()
    for i in range(len(WiC)):
        data = processWiC(WiC[i])
        words.append(data["word"])
        sentences1.append(data["sentence1"])
        sentences2.append(data["sentence2"])
        POSs.append(data["POS"])
        tags.append(gold[i])    
    useModels(llm,tokenizer,sb,filenameTest,words,sentences1,sentences2,POSs,tags,modelname,fewN, fewV)
    dotvalue = estimate(filenameTest, regrDot, "dot")
    dotvalueE = estimate(filenameTest, regrDotE, "dotE")
    file = open(filenameResult, "w",encoding='utf-8')
    print(dotvalue)
    file.write("Definition: " +str(dotvalue) + "\n")
    file.write("Definition + Context: " +str(dotvalueE) + "\n")
    file.close()
    
    
