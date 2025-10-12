import json

#Few False or a dictionary with k, words, examples and definitions to do the few-shot prompt.
def generate_prompt(modelname, word, example, few = {"k":0}):
    f = open('modelsData.json')
    data = json.load(f)[modelname]
    if data["type"] == "Instruct":
        init = ""
        for i in range(few["k"]):
            few = data["prompt"]["inst_start"]+"As an expert English lexicographer generate a dictionary definition of the word '" + few["words"][i] + "' in the sense of this example '" + few["examples"][i][0] + "'. Give JUST the definition not other things." + data["prompt"]["inst_end"]
            init += few + data["prompt"]["response_start"] + few["definitions"][i] + data["prompt"]["response_end"]
        last = data["prompt"]["inst_start"]+"As an expert English lexicographer generate a dictionary definition of the word '" + word + "' in the sense of this example '" + example + "'. Give JUST the definition not other things." + data["prompt"]["inst_end"]
        prompt = init + last + data["prompt"]["response_start"]
        return prompt
    elif data["type"] == "Chat":
        init = data["prompt"]["init_start"] + "You are an expert English lexicographer, generate a dictionary definition of a word given some example sentences of the word. Please, JUST provide the definition, not further explanation." + data["prompt"]["init_end"]
        for i in range(few["k"]):
            user = data["prompt"]["user_start"] + "Giving the word '" + few["words"][i] + "' and the sense of this example: '" + few["examples"][i][0] + "', generate the definition of the word in this sense. Give JUST the definition not further explanation." + data["prompt"]["user_end"]
            init += user + data["prompt"]["response_start"] + few["definitions"][i] + data["prompt"]["response_end"]
        user = data["prompt"]["user_start"] + "Giving the word '" + word + "' and the sense of this example: '" + example + "', generate the definition of the word in this sense. Give JUST the definition not further explanation." + data["prompt"]["user_end"]
        prompt = init + user + data["prompt"]["response_start"]
        return prompt
    else:
        #Only chat or instruct models
        assert False



def generate_promptV2(modelname, tokenizer,word, example, pos,fewN, fewV, tokenize=False):
    if pos == "N":
        few = fewN
        word_type = "noun"
    else: 
        few = fewV
        word_type = "verb"
    f = open('modelsData.json')
    data = json.load(f)[modelname]
    if data["type"] == "Instruct":
        chat = []
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "As an expert English lexicographer generate a dictionary definition of the "+ word_type + " '" + few["words"][i] + "' in the sense of this example '" + few["examples"][i][0] + "'. Give JUST the definition not other things."})
            chat.append({"role": "assistant", "content": few["definitions"][i]})
        chat.append({"role": "user", "content": "As an expert English lexicographer generate a dictionary definition of the "+ word_type + " '" + word + "' in the sense of this example '" + example + "'. Give JUST the definition not other things."})
    elif data["type"] == "Chat":
        chat = [{"role": "system", "content": "You are an expert English lexicographer, generate a dictionary definition of a word given some example sentences of the word. Please, JUST provide the definition, not further explanation."}]
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "Giving the "+ word_type + " '" + few["words"][i] + "' and the sense of this example: '" + few["examples"][i][0] + "', generate the definition of the word in this sense. Give JUST the definition not further explanation."})
            chat.append({"role": "assistant", "content": few["definitions"][i]})
        chat.append({"role": "user", "content": "Giving the "+ word_type + " '" + word + "' and the sense of this example: '" + example + "', generate the definition of the word in this sense. Give JUST the definition not further explanation."})
    
    else:
        assert False
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=tokenize)
    return prompt



def enumerate_examples(examples):
    enumerated=""
    for i in range(len(examples)):
        enumerated+=str(i+1)+". "+examples[i]
        if examples[i][-1] != ".":
            enumerated+="."
        enumerated += "\n"
    return enumerated[:-1]
    


def generate_promptExamples(modelname, tokenizer, word, example, pos, kE = 8, fewN = {"k":0}, fewV = {"k":0}):
    if pos == "N":
        few = fewN
        word_type = "noun"
    else: 
        few = fewV
        word_type = "verb"
    f = open('modelsData.json')
    k = str(kE)
    data = json.load(f)[modelname]
    if data["type"] == "Instruct":
        chat = []
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "As an expert English lexicographer generate and enumerate "+k+" examples of usage of the "+ word_type + " '" + few["words"][i] + "' in the sense of this example '" + few["examples"][i][0] + "'. Give JUST the examples, not other things."})
            chat.append({"role": "assistant", "content": enumerate_examples(few["examples"][i][1:])})

        chat.append({"role": "user", "content": "As an expert English lexicographer generate and enumerate "+k+" examples of the "+ word_type + " '" + word + "' in the sense of this example '" + example +"'. Give JUST the examples, not other things."})
    elif data["type"] == "Chat":
        chat = [{"role": "system", "content": "You are an expert English lexicographer, generate and enumerate "+ k +" usage examples of a word given another example sentences of that word. Give JUST the examples, not further explanation."}]
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "Giving the "+ word_type + " '" + few["words"][i] + "' and the sense of this example: '" + few["examples"][i][0] + "', generate another "+ k +" usage examples."})
            chat.append({"role": "assistant", "content": enumerate_examples(few["examples"][i][1:])})
        chat.append({"role": "user", "content": "Giving the "+ word_type + " '" + word + "' and the sense of this example: '" + example + "', generate another "+ k +" usage examples."})
    else:
        assert False
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_promptExampleDef(modelname, tokenizer, word, definition, pos, fewN = {"k":0}, fewV = {"k":0}, tokenize=False):
    if pos == "N":
        few = fewN
        word_type = "noun"
    else: 
        few = fewV
        word_type = "verb"
    f = open('modelsData.json')
    data = json.load(f)[modelname]
    if data["type"] == "Instruct":
        chat = []
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "As an expert English lexicographer generate ONLY ONE example of the usage of the "+ word_type + " '" + few["words"][i] + "' in the sense of this definition: '" + few["definitions"][i] + "'. Provide JUST the example NOT include the definition or any other further explanations."})
            chat.append({"role": "assistant", "content": few["examples"][i][0]})
        
        chat.append({"role": "user", "content": "As an expert English lexicographer generate ONLY ONE example of usage of the "+ word_type + " '" + word + "' in the sense of this definition: '" + definition + "'. Provide JUST the example NOT include the definition or any other further explanations."})
    elif data["type"] == "Chat":
        chat = [{"role": "system", "content": "You are an expert English lexicographer, generate ONLY ONE example of the usage of a word given a definition of that word. Provide JUST the example NOT include the definition or any other further explanations."}]
        for i in range(few["k"]):
            chat.append({"role": "user", "content": "Giving the "+ word_type + " '" + few["words"][i] + "' and the sense of this definition: '" + few["definitions"][i] + "', generate one usage example."})
            chat.append({"role": "assistant", "content": few["examples"][i][0]})
        chat.append({"role": "user", "content": "Giving the "+ word_type + " '" + word + "' and the sense of this definition: '" + definition + "', generate one usage example."})
    else:
        assert False
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=tokenize)
    return prompt