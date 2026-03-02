import json 
import re
from tqdm import tqdm


def processOxford(data,addsubdefs = False):
    examples = []
    defin = ""
    refn = ""
    difG = 0
    pos = ""
    oxf = []
    for i in tqdm(range(len(data))):
        if i + difG >= len(data):
            break
        row = data[i + difG]
        if row[0:6] == "LETTER":
            defn = 0
            refn = ""
            word = re.findall("WORD: [^:]+",row)[0][6:]
            word = word.replace("_"," ")
            word = word.replace("\t","")
            #i+1 tratamendua
            difG += 1
            row = data[i+difG]
            while "\tDEF:" not in row and "\tREF:" not in row:
                difG +=1
                row = data[i+difG]
            if "\tDEF:" in row:
                defn = int(re.findall("DEF: [0-9]+",row)[0][5:])
                if defn > 9:
                    luz = 8
                else:
                    luz = 7
                defin = re.findall("DEF: .*",row)[0][luz:]
                defin = defin.replace("","")
                pos = re.findall("POS: [^:]+",row)[0][5:-4]
                examples, dif = searchExamples(data,i+difG,refn,defn,pos,addsubdefs)
                difG += dif
                enter = {"word": word, "POS": pos,"definition": defin, "examples": examples}
                oxf.append(enter)
                
            else:
                refn = get_ref(row)
                refn = refn.replace("_"," ")
                refn = refn.replace("\t","")
                pos = re.findall("POS: [^:]+",row)[0][5:-4]
                examples, dif = searchExamples(data,i+difG,refn,defn,pos,addsubdefs)
                difG += dif

                enter = {"word": word, "POS": pos,"definition": "$REF:$"+refn, "examples": examples}
                oxf.append(enter)
        elif row == "" or row[:3] == "EF:":
            continue
        elif "\tDEF:" in row:
            defn = int(re.findall("DEF: [0-9]+",row)[0][5:])
            refn = ""
            if defn > 9:
                luz = 8
            else:
                luz = 7
            defin = re.findall("DEF: .*",row)[0][luz:]
            defin = defin.replace("","")
            pos = re.findall("POS: [^:]+",row)[0][5:-4]
            examples, dif = searchExamples(data,i+difG,refn,defn,pos,addsubdefs)
            difG += dif
            enter = {"word": word, "POS": pos,"definition": defin, "examples": examples}
            oxf.append(enter)
        elif "\tREF:" in row:
            refn = get_ref(row)
            refn = refn.replace("_"," ")
            refn = refn.replace("\t","")
            defn = 0
            pos = re.findall("POS: [^:]+",row)[0][5:-4]
            examples, dif = searchExamples(data,i+difG,refn,defn,pos,addsubdefs)
            difG += dif
            enter = {"word": word, "POS": pos,"definition": "$REF:$"+refn, "examples": examples}
            oxf.append(enter)
    return oxf

def searchExamples(data,ind,refn,defn, pos,addsubdefs):
    examples = []
    j = ind + 1
    for j in range(ind+1,len(data)):

        row = data[j]
        if "DEF:" in row:
            defnb = int(re.findall("DEF: [0-9]+",row)[0][5:])
        else:
            defnb = defn
        if "REF:" in row:
            refnb = get_ref(row)
        else:
            refnb = refn
        if "POS:" in row:
            posb = re.findall("POS: [^:]+",row)[0][5:-4]
        else:
            posb = pos
                    
        if "SUBDEF:" in row and "EX:" not in row and not addsubdefs:
            continue
        elif "SUBDEF:" in row and addsubdefs:
            print("TODO")
            #TODO Subdef-en tratamendua hemen in
        elif row == "" or "SYN:" in row or "DOM:" in row:
            continue
        elif ("EX:" not in row and (refnb != refn or pos != posb or defnb != defn)) or "LETTER:" in row:
            break
        elif "EX:" in row:
            exampleA = re.split("EX: [0-9][^a-zA-Z]",row)[1]
            exampleB = exampleA.replace("\t","")
            exampleC = exampleB.replace("‘","")
            exampleD = exampleC.replace("’","")
            example = exampleD.replace("","")
            examples.append(example)
        elif refnb == refn:
            #print("HEMEN")
            continue
        else:
            assert False
    dif = j-ind-1
    return examples, dif


            

            
def get_ref(row):

    if "EX:" in row:
        ref = re.findall(r"REF:\s*([^:]+)", row)[0][:-2]
        ref = ref.replace("_"," ")
        ref = ref.replace("\t","")
        if ref[0] == " ":
            ref = ref[1:]
    else:
        ref = re.findall(r"REF:\s*([^\n]+)", row)[0]
    return ref

def recursion(data, word, pos,preword):
    for row in data:
        if str.upper(row["word"]) == str.upper(word) and str.upper(pos) == str.upper(row["POS"]):
            if str.upper(row["word"]) in preword:
                return "Circular definition $REF:$"+word
            
            if row["definition"][:6] == "$REF:$":
                ref = row["definition"][6:]
                preword.append(str.upper(row["word"]))
                return recursion(data, ref, row["POS"], preword)
            else:
                return row["definition"]
    return "Definition not found $REF:$"+word


def recursive_def(data):
    writeF = open("CorpusOxford.json", "w", encoding='utf-8')
    for row in tqdm(data):
        if row["definition"][:6] == "$REF:$":
            ref = row["definition"][6:]
            pos = row["POS"]
            definition = recursion(data,ref,pos,[str.upper(row["word"])])
            row["definition"] = definition
        writeF.write(json.dumps(row, ensure_ascii=False) + "\n")
    writeF.close()




if __name__ == "__main__":
    file = open("Oxford/oxford.txt", "r")
    dataO = file.read().split("\n")
    file.close()
    print("Processing Oxford Corpus...")
    oxf = processOxford(dataO)
    print("Preprocess finished, getting recursive definitions...")
    recursive_def(oxf)
    print("Recursive definitions finished and saved in CorpusOxford.json")
    
