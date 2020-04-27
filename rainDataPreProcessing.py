#!/usr/bin/env python
# coding: utf-8


import shutil
import os
import numpy as np
import argparse
from os import path


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)



dirName = "data/rain/rainy-image-dataset-master/"
gtPath = dirName + "ground truth"
gtTrainPath = dirName + "ground_truth_train"
gtTestPath = dirName+ "ground_truth_test"

riTrainPath = dirName + "rainy_image_train"
riTestPath = dirName + "rainy_image_test"
riPath = dirName + "rainy image"

os.mkdir(gtTrainPath)
os.mkdir(gtTestPath)
os.mkdir(riTrainPath)
os.mkdir(riTestPath)



x = get_files_from_folder(gtPath)

SPLIT_RATIO = 0.8

numTotalFiles = x.shape[0]
numTrainFiles = int(numTotalFiles* SPLIT_RATIO)
numTestFiles = numTotalFiles - numTrainFiles

gtTrainFiles = np.random.choice(x, numTrainFiles, replace = False)


allfiles = x.tolist()
for i in range(numTrainFiles):
    fileName = gtTrainFiles[i]
    allfiles.remove(fileName)

gtTestFiles = allfiles


riTrainFiles = []

for file in gtTrainFiles:
    a,b = file.split(".")
    for i in range(1, 15):
        newName = ""
        newName = a+"_"+ str(i)+ "." +b
        riTrainFiles.append(newName)
        

riTestFiles = []

for file in gtTestFiles:
    a,b = file.split(".")
    for i in range(1, 15):
        newName = ""
        newName = a+"_"+ str(i)+ "." +b
        riTestFiles.append(newName)
        



destination = gtTrainPath
for file in gtTrainFiles:
    source = gtPath + "/"+ file
    print(source)
    dest = shutil.copy(source, destination)


destination = gtTestPath    
for file in gtTestFiles:
    source = gtPath + "/"+ file
    print(source)
    dest = shutil.copy(source, destination)
    


destination = riTrainPath    
for file in riTrainFiles:
    source = riPath + "/"+ file
    print(source)
    dest = shutil.copy(source, destination)   

destination = riTestPath    
for file in riTestFiles:
    source = riPath + "/"+ file
    print(source)
    dest = shutil.copy(source, destination)   