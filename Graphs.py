import csv
import os
from matplotlib import pyplot as plt

def visualizeData():
    #Read all the csv files, merge it all into data
    data = [] #This is what is passed into the api
    labels =[]
    inivData = [] # This looks liek

    curDir = os.getcwd()
    files = os.listdir(curDir+"/csv_files")
    filePaths = [curDir+'/csv_files/'+file for file in files]

    for x, filePath in enumerate(filePaths):
        with open(filePath) as csvfile:
            datum = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(datum):
                if i != 0:
                    data.append(row)
                    labels.append(x)
                if i >= 2000: #If more than 2000 datapoints, breakout. We only take first 2000.
                    break
        inivData.append(data)



    # Data

    #Set the labels

    for k in range(2, 6):
        plt.xlabel("Time (s)")
        plt.grid(linestyle="dotted")
        plt.title("Time v. " + "Channel " + str(k-1))
        plt.ylabel("Channel " + str(k-1))
        plt.plot([i for i in range(0,len(data), 40)], [data[x][k] for x in range(0, len(data), 40)])
        plt.show()


if __name__ == '__main__':
    pass

