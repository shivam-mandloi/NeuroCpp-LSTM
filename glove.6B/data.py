def ReadData(path):
    data = ""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def WriteData(data, fileName):    
    with open(fileName, "w", encoding="utf-8") as f:
        f.write(data)    


data = ReadData(r'C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NeuroCpp-LSTM\glove.6B\temp.txt')

data = data.split("\n\n")

print(data[0])