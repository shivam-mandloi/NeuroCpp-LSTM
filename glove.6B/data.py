def ReadData(path):
    data = ""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def WriteData(data, fileName):    
    with open(fileName, "w", encoding="utf-8") as f:
        f.write(data)    


data = ReadData(r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroVec-RNN\dataset\poem.txt").lower()
WriteData(data, "poem.txt")
li = data.split("\n")

words = dict()

for sent in li:
    splitedWords = sent.split(" ")
    for word in splitedWords:
        words[word] = 0
print("[*] Get all the words")
data = ReadData(r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroVec-RNN\glove.6B\glove.6B.300d.txt").split("\n")


dataStore = ""
index = 0
for vector in data:
    vec = vector.split(" ")
    if vec[0] in words:
        dataStore += vector + "\n"
        words[vec[0]] = 1

print("[*] Get all the vector")

for word in words.keys():
    if(words[word] == 0):
        dataStore +=  word + " " + ' '.join(["0"] * 300) + "\n"

data = ""
WriteData(dataStore, "GetRelatedData.txt")