import shutil
import os
import numpy as np
import argparse
from os import path

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)


def renamePath(path, addon):
    x = path.split("/")
    x[-1]= addon +  x[-1]
    
    result =""
    for i in range(len(x)-1):
        result = result + x[i] + "/"
    
    result = result + x[-1].replace(" ", "") 
    return result

def copyfiles(fileNames, filePath, destination):
    for file in fileNames:
        source = filePath + "/"+ file
        if path.exists(source):
            dest = shutil.copy(source, destination)
    

def copyfilesWithRename(fileNames, filePath, destination, addon):
    for file in fileNames:
        source = filePath + "/"+ file
        if path.exists(source):
            dest = shutil.copy(source, destination)
            dest = dest.replace("\\", "/")
            final = renamePath(dest, addon)
            os.rename(dest, final)
            
dirName = "data/reflection/SIR/"
postCardDatasetName = "PostcardDataset_Flat"
solidObjectDatasetName = "SolidObjectDataset_Flat"
wildSceneDatasetName = "WildSceneDataset_Flat"


gtTrainPath = dirName + "ground_truth_train"
gtTestPath = dirName+ "ground_truth_test"

mixTrainPath = dirName + "mixed_image_train"
mixTestPath = dirName + "mixed_image_test"

pcFilePath = dirName + postCardDatasetName
soFilePath = dirName + solidObjectDatasetName
wsFilePath = dirName + wildSceneDatasetName

SPLIT_RATIO = 0.8


print("Creating Following Directories:")
print(gtTrainPath)
os.mkdir(gtTrainPath)

print(gtTestPath)
os.mkdir(gtTestPath)

print(mixTrainPath)
os.mkdir(mixTrainPath)

print(mixTestPath)
os.mkdir(mixTestPath)


print("Spliting PostCard Dataset")

pcFiles = get_files_from_folder(pcFilePath)

pcGt = []
for file in pcFiles:
    if "-g-" in file:
        mix =  file.replace("-g-", "-m-", 1)
        if mix in pcFiles:
            pcGt.append(file)
        else:
            print("We have a problem  ", file, mix)

            
numGtFiles = len(pcGt)
numGtTrainFiles = int(numGtFiles * SPLIT_RATIO)
numGtTestFiles = numGtFiles - numGtTrainFiles

gtTrainFiles = np.random.choice(pcGt, numGtTrainFiles, replace = False)

allfiles = pcGt
for i in range(numGtTrainFiles):
    fileName = gtTrainFiles[i]
    allfiles.remove(fileName)

gtTestFiles = allfiles
mixTestFiles = []
mixTrainFiles =[]

for file in gtTestFiles:
    mixTestFiles.append(file.replace("-g-", "-m-", 1))
    
for file in gtTrainFiles.tolist():
    mixTrainFiles.append(file.replace("-g-", "-m-", 1))
    
copyfiles(gtTrainFiles, pcFilePath, gtTrainPath)
copyfiles(gtTestFiles, pcFilePath, gtTestPath)
copyfiles(mixTrainFiles, pcFilePath, mixTrainPath)
copyfiles(mixTestFiles, pcFilePath, mixTestPath)
print("PostCard Dataset Copied")



print("Spliting SolidObject Dataset")
soFiles = get_files_from_folder(soFilePath)

soGt = []
for file in soFiles:
    if "g " in file:
        mix =  file.replace("g ", "m ", 1)
        if mix in soFiles:
            soGt.append(file)
        else:
            print("We have a problem  ", file, mix)
            
numGtFiles = len(soGt)
numGtTrainFiles = int(numGtFiles * SPLIT_RATIO)
numGtTestFiles = numGtFiles - numGtTrainFiles

gtTrainFiles = np.random.choice(soGt, numGtTrainFiles, replace = False)
allfiles = soGt

for i in range(numGtTrainFiles):
    fileName = gtTrainFiles[i]
    allfiles.remove(fileName)

gtTestFiles = allfiles
mixTestFiles = []
mixTrainFiles =[]

for file in gtTestFiles:
    mixTestFiles.append(file.replace("g ", "m ", 1))
    
for file in gtTrainFiles.tolist():
    mixTrainFiles.append(file.replace("g ", "m ", 1))

copyfilesWithRename(gtTrainFiles, soFilePath, gtTrainPath, "so_")
copyfilesWithRename(gtTestFiles, soFilePath, gtTestPath, "so_")
copyfilesWithRename(mixTrainFiles, soFilePath, mixTrainPath, "so_")
copyfilesWithRename(mixTestFiles, soFilePath, mixTestPath, "so_")
print("SolidObject Dataset Copied")


print("Spliting Wild Scene Dataset")

wsFiles = get_files_from_folder(wsFilePath)

wsGt = []
for file in wsFiles:
    if "g " in file:
        mix =  file.replace("g ", "m ", 1)
        if mix in wsFiles:
            wsGt.append(file)
        else:
            print("We have a problem  ", file, mix)
            
numGtFiles = len(wsGt)
numGtTrainFiles = int(numGtFiles * SPLIT_RATIO)
numGtTestFiles = numGtFiles - numGtTrainFiles

gtTrainFiles = np.random.choice(wsGt, numGtTrainFiles, replace = False)
allfiles = wsGt

for i in range(numGtTrainFiles):
    fileName = gtTrainFiles[i]
    allfiles.remove(fileName)

gtTestFiles = allfiles
mixTestFiles = []
mixTrainFiles =[]

for file in gtTestFiles:
    mixTestFiles.append(file.replace("g ", "m ", 1))
    
for file in gtTrainFiles.tolist():
    mixTrainFiles.append(file.replace("g ", "m ", 1))

    
copyfilesWithRename(gtTrainFiles, soFilePath, gtTrainPath, "ws_")
copyfilesWithRename(gtTestFiles, soFilePath, gtTestPath, "ws_")
copyfilesWithRename(mixTrainFiles, soFilePath, mixTrainPath, "ws_")
copyfilesWithRename(mixTestFiles, soFilePath, mixTestPath, "ws_")
print("Wild Scene Dataset Copied")
print("Done!")