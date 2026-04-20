import numpy as np
import csv
import matplotlib.pyplot as plt

X = np.load("clear_segments.npy", allow_pickle=True)

ecg = X[:,1,:]
ppg = X[:,0,:]  

print(ppg.shape)
print(ecg.shape)

for i in range(ppg.shape[0]):

    with open(f"./data/ppg/ppg_{i+1}.csv",'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["A1"])
        for item in ppg[i]:
            writer.writerow([item])

    with open(f"./data/ecg/ecg_{i+1}.csv",'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["A1"])
        for item in ecg[i]:
            writer.writerow([item])

#visualize
figure = plt.figure(figsize=(10, 5))
plt.plot(list(range(250)), ppg[0], label = "PPG")
plt.plot(list(range(250)), ecg[0], label = "ECG")
plt.legend()
plt.show()