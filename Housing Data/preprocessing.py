import pandas as pd
import numpy as np

def Write(data, filename = ""):
    with open(filename, "w") as f:
        f.write(data)

data = pd.read_csv("housing.csv")
li = []
for da in data:
    li.append(da)

map = dict()
dataToStore = ""
targetTostore = ""
index = 0

for i in data[li[-1]]:
    if i not in map:
        map[i] = index
        index += 1
for index, row in data.iterrows():
    temp = []
    check = False
    for head in li[:-2]:
        temp.append(str(row[head]))
        if str(row[head]) == 'nan':
            check = True
            break
    if(check): continue
    targetTostore += str(row[-2]) + "\n"
    temp.append(str(map[row[li[-1]]]))
    dataToStore += (' '.join(temp) + "\n")

print("[#] Store Data")
Write(dataToStore, r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NueroCpp-FNN\Housing Data\data.txt")
Write(targetTostore, r"C:\Users\shiva\Desktop\IISC\code\NeuroCpp\NueroCpp-FNN\Housing Data\target.txt")

data = np.loadtxt("data.txt")

index = np.array([i for i in range(len(data))])
np.random.shuffle(index)
data = data[index]

trainIndex = int(len(data) * 0.7)

trainData = data[:trainIndex]
testData = data[trainIndex:]
np.savetxt("trainData.txt",trainData)
np.savetxt("testData.txt",testData)

target = np.loadtxt("target.txt")
target = target[index]
trainTargetData = target[:trainIndex]
testTargetData = target[trainIndex:]
np.savetxt("trainTarget.txt", trainTargetData)
np.savetxt("testTarget.txt", testTargetData)