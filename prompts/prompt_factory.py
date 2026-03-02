from abc import abstractmethod
import json

class Prompt():
    def __init__(self, language):
        self.language = language

    def generate_promptV2(self, modelname, tokenizer, word, example, pos, fewN, fewV, tokenize=False):
        if pos == "noun" or pos == "sustantivo" or pos == "izena" or pos == "N":
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
                chat.append({"role": "user", "content": "As an expert " + self.language + " lexicographer, generate a dictionary definition in " + self.language + " of the "+ word_type + " '" + few["words"][i] + "' in the sense of this example '" + few["examples"][i][0] + "'. Give JUST the definition without further explanation."})
                chat.append({"role": "assistant", "content": few["definitions"][i]})
            chat.append({"role": "user", "content": "As an expert " + self.language + " lexicographer, generate a dictionary definition in " + self.language + " of the "+ word_type + " '" + word + "' in the sense of this example: '" + example + "'. Give JUST the definition without further explanation."})
        elif data["type"] == "Chat":
            chat = [{"role": "system", "content": "You are an expert " + self.language + " lexicographer. Your task is to generate a dictionary definition in " + self.language + " of a word, given some example sentences of the word. Please, provide JUST the definition —DO NOT include the example or any other further explanations."}]
            for i in range(few["k"]):
                chat.append({"role": "user", "content": "Given the "+ word_type + " '" + few["words"][i] + "' and its sense in this example: '" + few["examples"][i][0] + "', generate the definition of the word for that sense. Give JUST the definition without further explanation."})
                chat.append({"role": "assistant", "content": few["definitions"][i]})
            chat.append({"role": "user", "content":  "Given the "+ word_type + " '" + word + "' and its sense in this example: '" + example + "', generate the definition of the word for that sense. Give JUST the definition without further explanation."})
        
        else:
            assert False
        prompt = tokenizer.apply_chat_template(chat, tokenize=tokenize, add_generation_prompt=True)
        return prompt
    
    def generate_promptExampleDef(self, modelname, tokenizer, word, definition, pos, fewN = {"k":0}, fewV = {"k":0}, tokenize=False):
        if pos == "noun" or pos == "sustantivo" or pos == "izena" or pos == "N":
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
                chat.append({"role": "user", "content": "As an expert " + self.language + " lexicographer, generate ONLY ONE example in " + self.language + " of the usage of the "+ word_type + " '" + few["words"][i] + "', in the sense of this definition: '" + few["definitions"][i] + "'. Provide JUST the example—DO NOT include the definition or any other further explanation."})
                chat.append({"role": "assistant", "content": few["examples"][i][0]})

            chat.append({"role": "user", "content": "As an expert " + self.language + " lexicographer, generate ONLY ONE example in " + self.language + " of the usage of the "+ word_type + " '" + word + "', in the sense of this definition: '" + definition + "'. Provide JUST the example—DO NOT include the definition or any other further explanation."})
        elif data["type"] == "Chat":
            chat = [{"role": "system", "content": "You are an expert " + self.language + " lexicographer. Your task is to generate ONLY ONE example in " + self.language + " of the usage of a word, given a definition of that word. Please, provide JUST the example—DO NOT include the definition or any other further explanation."}]
            for i in range(few["k"]):
                chat.append({"role": "user", "content": "Given the "+ word_type + " '" + few["words"][i] + "' and its sense in this definition: '" + few["definitions"][i] + "', generate one usage example of the word for that sense. Give JUST the example without further explanation."})
                chat.append({"role": "assistant", "content": few["examples"][i][0]})
            chat.append({"role": "user", "content":  "Given the "+ word_type + " '" + word + "' and its sense in this definition: '" + definition + "', generate one usage example of the word for that sense. Give JUST the example without further explanation."})
        else:
            assert False
        prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=tokenize)
        return prompt

class EnglishPrompt(Prompt):

    def generate_promptV2(modelname, tokenizer,word, example, pos,fewN, fewV, tokenize = False):
        if pos == "N" or pos == "noun":
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
        prompt = tokenizer.apply_chat_template(chat, tokenize=tokenize, add_generation_prompt=True)
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

class SpanishPrompt(Prompt):

    def generate_promptV2(modelname, tokenizer, word, example, pos, fewN, fewV, tokenize = False):
        if pos == "N":
            few = fewN
            word_type_es = "sustantivo"
        else:
            few = fewV
            word_type_es = "verbo"

        f = open('modelsData.json')
        data = json.load(f)[modelname]

        if data["type"] == "Instruct":
            chat = []
            for i in range(few["k"]):
                chat.append({
                    "role": "user",
                    "content": (
                        "Como lexicógrafo experto en español, genera una definición de diccionario del "
                        + word_type_es + " '" + few["words"][i] + "' en el sentido de este ejemplo '"
                        + few["examples"][i][0] + "'. Proporciona SOLO la definición, sin nada más."
                    )
                })
                chat.append({"role": "assistant", "content": few["definitions"][i]})
            chat.append({
                "role": "user",
                "content": (
                    "Como lexicógrafo experto en español, genera una definición de diccionario del "
                    + word_type_es + " '" + word + "' en el sentido de este ejemplo '" + example
                    + "'. Proporciona SOLO la definición, sin nada más."
                )
            })

        elif data["type"] == "Chat":
            chat = [{
                "role": "system",
                "content": (
                    "Eres un lexicógrafo experto en español. Genera una definición de diccionario de una palabra a partir de un ejemplo de uso. Por favor, PROPORCIONA SOLO la definición, sin más explicación."
                )
            }]
            for i in range(few["k"]):
                chat.append({
                    "role": "user",
                    "content": (
                        "Dado el " + word_type_es + " '" + few["words"][i] +
                        "' y el siguiente ejemplo: '" + few["examples"][i][0] +
                        "', genera la definición del término en ese sentido. Proporciona SOLO la definición, sin más explicación."
                    )
                })
                chat.append({"role": "assistant", "content": few["definitions"][i]})
            chat.append({
                "role": "user",
                "content": (
                    "Dado el " + word_type_es + " '" + word + "' y el siguiente ejemplo: '"
                    + example + "', genera la definición del término en ese sentido. Proporciona SOLO la definición, sin más explicación."
                )
            })

        else:
            assert False

        prompt = tokenizer.apply_chat_template(chat, tokenize=tokenize, add_generation_prompt=True)
        return prompt
    
    def generate_promptExampleDef(modelname, tokenizer, word, definition, pos, fewN, fewV, tokenize=False):
        if pos == "N":
            few = fewN
            word_type_es = "sustantivo"
        else:
            few = fewV
            word_type_es = "verbo"

        f = open('modelsData.json')
        data = json.load(f)[modelname]

        if data["type"] == "Instruct":
            chat = []
            for i in range(few["k"]):
                chat.append({
                    "role": "user",
                    "content": (
                        "Como lexicógrafo experto en español, genera SOLO UN ejemplo de uso del "
                        + word_type_es + " '" + few["words"][i] + "' en el sentido de esta definición: '"
                        + few["definitions"][i] + "'. Proporciona SOLO el ejemplo, sin incluir la definición ni ninguna otra explicación."
                    )
                })
                chat.append({"role": "assistant", "content": few["examples"][i][0]})

            chat.append({
                "role": "user",
                "content": (
                    "Como lexicógrafo experto en español, genera SOLO UN ejemplo de uso del "
                    + word_type_es + " '" + word + "' en el sentido de esta definición: '" + definition
                    + "'. Proporciona SOLO el ejemplo, sin incluir la definición ni ninguna otra explicación."
                )
            })

        elif data["type"] == "Chat":
            chat = [{
                "role": "system",
                "content": (
                    "Eres un lexicógrafo experto en español. Genera SOLO UN ejemplo de uso de una palabra dada su definición. Proporciona SOLO el ejemplo, sin incluir la definición ni ninguna otra explicación."
                )
            }]
            for i in range(few["k"]):
                chat.append({
                    "role": "user",
                    "content": (
                        "Dado el " + word_type_es + " '" + few["words"][i] +
                        "' y el sentido de esta definición: '" + few["definitions"][i] +
                        "', genera un ejemplo de uso. Proporciona SOLO el ejemplo, sin más explicaciones."
                    )
                })
                chat.append({"role": "assistant", "content": few["examples"][i][0]})

            chat.append({
                "role": "user",
                "content": (
                    "Dado el " + word_type_es + " '" + word + "' y el sentido de esta definición: '"
                    + definition + "', genera un ejemplo de uso. Proporciona SOLO el ejemplo, sin más explicaciones."
                )
            })

        else:
            assert False

        prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=tokenize)
        return prompt

def get_promptFactory(language):
    if language == "EN":
        return Prompt(language="English")
    elif language == "ES":
        return Prompt(language="Spanish")
    elif language == "EU":
        return Prompt(language="Basque")