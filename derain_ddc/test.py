# coding: utf-8
from keras.models import *
import zf_unet_224_model
import data_generator as dg
model = zf_unet_224_model.ZF_UNET_224()
model.load_weights("zf_unet_224_epcho260.h5")

#model = load_model("zf_unet_224_dust_ori.h5")

import numpy as np
import cv2
import os
import time
import math
from time import time

import matplotlib.pyplot as plt

from keras import backend as K

image_list = []
dg.batch_generator(1)

def preprocess(image):
    print image.shape



    h,w,c = image.shape
    y_ratio = int(math.ceil(h/224.0))
    x_ratio = int(math.ceil(w/224.0))
    print x_ratio,y_ratio
    x_res_full= 224 * x_ratio
    y_res_full= 224 * y_ratio
    x_padding = int((x_res_full - w)/2)
    y_padding = int((y_res_full - h)/2)

    image = cv2.copyMakeBorder(image,y_padding,y_padding,x_padding,x_padding,cv2.BORDER_REFLECT)
    padding = [y_padding, y_padding, x_padding, x_padding]
    return cv2.resize(image,(x_res_full,y_res_full)),padding

def removePadding(image,padding):
    shapex = image.shape
    y_padding,y_padding,x_padding,x_padding = padding
    return image[y_padding:shapex[0]-y_padding-1,x_padding:shapex[1]-x_padding]



def processAll(name):
    X,padding = preprocess(cv2.imread(name))
    input_image = X.astype(np.float)/255

    start = time()

    pre = model.predict(np.array([input_image]))

    stop = time()
    print("Stop: " + str(stop))
    print(str(stop-start) )

    pred = pre[0]*255
    output_image = pred.astype(np.uint8)
    input_image = input_image.astype(np.uint8)
    output_image = removePadding(output_image,padding)
    cv2.imwrite("output.png",output_image)
    cv2.imshow("output",output_image)
    cv2.waitKey(0)


path = "/home/syan/workspace/rainy_image_dataset/test"
images = [] 
for root, dirs, files in os.walk(path):     
    for f in files :
        images.append(f)
#for i in range(len(images)):
    #processAll(path + "/" + images[i])


processAll("real_rain_img/re-1.jpg")
processAll("real_rain_img/2-r.jpg")



