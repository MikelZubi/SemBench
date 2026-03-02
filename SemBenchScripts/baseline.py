from sentence_transformers import SentenceTransformer, util
import json 
from tqdm import tqdm
import os





def testModels(word, example, POS, label, def1,def2, sb, file):


    
    embeddingE = sb.encode(example, convert_to_tensor=True)
    embedding1 = sb.encode(def1, convert_to_tensor=True)
    embedding2 = sb.encode(def2, convert_to_tensor=True)


    dot_score1 = util.dot_score(embeddingE, embedding1)[0][0].item()
    dot_score2 = util.dot_score(embeddingE, embedding2)[0][0].item()
    if dot_score1 > dot_score2:
        pred_label = 0
        dot_score = dot_score1
    else:
        pred_label = 1
        dot_score = dot_score2
    dictionary = {"word": word, "POS": POS, "example": example, "def1": def1 , "def2": def2, "dot": dot_score, "label": label, "pred_label": pred_label}
    file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def useModels(sb,filename,words, examples, POSs, labels, def1s,def2s):

    file = open(filename, "w",encoding='utf-8')
    for i in tqdm(range(len(words))):
        testModels(words[i], examples[i],POSs[i], labels[i], def1s[i], def2s[i],sb, file)
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
    #pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id # Hack to fix a bug in transformers (batch_size)
    dificulties = ["easy", "medium", "hard", "random"]
    words = {"easy":[], "medium":[], "hard":[], "random":[]}
    examples = {"easy":[], "medium":[], "hard":[], "random":[]}
    POSs = {"easy":[], "medium":[], "hard":[], "random":[]}
    labels = {"easy":[], "medium":[], "hard":[], "random":[]}
    def1s = {"easy":[], "medium":[], "hard":[], "random":[]}
    def2s = {"easy":[], "medium":[], "hard":[], "random":[]}
    filename = "WSDOutputs/Baseline/test.json"
    sb = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    dotvalues = {}
    for dificultie in dificulties:
        with open("WSD/test_"+dificultie+".json","r") as Wsd_data:
            Wsd = Wsd_data.read().splitlines()
            for i in range(len(Wsd)):
                data = json.loads(Wsd[i])
                words[dificultie].append(data["word"])
                examples[dificultie].append(data["example"])
                POSs[dificultie].append(data["POS"])
                labels[dificultie].append(data["label"])
                def1s[dificultie].append(data["definitions"][0])
                def2s[dificultie].append(data["definitions"][1])
        

        
        useModels(sb,filename,words[dificultie], examples[dificultie], POSs[dificultie], labels[dificultie], def1s[dificultie], def2s[dificultie])
        dotvalues[dificultie] = estimate(filename)
    filenameResult = "WSDOutputs/Baseline/results.txt"
    file = open(filenameResult, "w",encoding='utf-8')
    for dificulti in dificulties:
        file.write(dificulti + ": " + str(dotvalues[dificulti]) + "\n")
    
    
