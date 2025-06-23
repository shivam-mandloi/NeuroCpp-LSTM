import re
import json

def ReadTxt(path):
    with open(path, "r") as f:
        data = f.read()
    return data

def SaveFile(path, data):
    with open(path, "w") as f:
        data = f.write(data)    

path = r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroVec-RNN\dataset\limerick_dataset_oedilf_v3.json"

with open(path, "r") as f:
    data = json.load(f)

exp = r"[;,'?\"()—_]|\.+"
storeData = ""
print(len(data))
for sentence in data[:10000]:
    storeData += re.sub(exp, "", sentence["limerick"]) + "\n" + "||" + "\n"

path = r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroVec-RNN\dataset\poem.txt"
SaveFile(path, storeData)

data = ReadTxt(r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroVec-RNN\dataset\poem.txt")

words = re.split(r"[ \n]", data)
allWords = dict()
for word in words:
    if(word == "|" or word == "||" or word == "\n"):
        continue
    allWords[word] = 1

words = ' '.join(allWords.keys())

SaveFile(r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroVec-RNN\dataset\allWords.txt", words)