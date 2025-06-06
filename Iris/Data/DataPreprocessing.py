import numpy as np

data = np.loadtxt(r"C:\Users\shiva\Desktop\IISC\temp\New folder (2)\Data\Iris.txt")
np.random.shuffle(data)
print(data)

np.savetxt("Iris.txt", data)