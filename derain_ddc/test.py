# coding: utf-8
from keras.models import *
import zf_unet_224_model
import data_generator as dg
model = zf_unet_224_model.ZF_UNET_224()
model.load_weights("temp.h5")

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

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1))))

def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def preprocess(image):
    #print image.shape



    h,w,c = image.shape
    y_ratio = int(math.ceil(h/224.0))
    x_ratio = int(math.ceil(w/224.0))
    #print x_ratio,y_ratio
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



def processAll(input_name, output_name, flag):
    print("Entering Process ALL: "+input_name)
    X,padding = preprocess(cv2.imread(input_name))
    Y_true, pad = preprocess(cv2.imread(output_name))
    print("Pre process done")
    input_image = X.astype(np.float)/255
    gt_image = Y_true.astype(np.float)/255
    print("Conv done")
    start = time()

    pre = model.predict(np.array([input_image]))
    print("Predict done")
    stop = time()
    print("Stop: " + str(stop))
    print(str(stop-start) )

    pred = pre[0]*255
    output_image = pred.astype(np.uint8)
    input_image = input_image.astype(np.uint8)
    output_image = removePadding(output_image,padding)
    if flag == 0:
        cv2.imwrite("/u/eot/manavm3/ORAs/outputs/reflection/"+input_name+".png",output_image)
    
    elif flag == 1:
        cv2.imwrite("/u/eot/manavm3/ORAs/outputs/rain/"+input_name+".png",output_image)
    
    elif flag ==  2:
        cv2.imwrite("/u/eot/manavm3/ORAs/outputs/reflection/"+input_name+".png",output_image)
    return PSNR(gt_image, pre[0]), SSIM(gt_image, pre[0]) 


path = ["../data/reflection/SIR/mixed_image_test","../data/rain/rainy-image-dataset-master/rainy_image_test","../data/de-fencing/SynthesizedData/test/fency"]
gt_path = ["../data/reflection/SIR/ground_truth_test","../data/rain/rainy-image-dataset-master/ground_truth_test","../data/de-fencing/SynthesizedData/test/gt"]
images = []
gt_images = []
sum_psnr = 0.0
sum_ssim = 0.0
for i in range(len(path)):
    for root, dirs, files in os.walk(path[i]):     
        for f in files :
            images.append(f)
for i in range(len(gt_path)):
    for root, dirs, files in os.walk(gt_path[i]):
        for f in files:
            gt_images.append((f,i))
for i in range(len(images)):
    psnr, ssim = processAll(path + "/" + images[i], gt_path + "/" + gt_images[i][0], gt_images[i][1])
    sum_psnr = sum_psnr + psnr
    sum_ssim = sum_ssim + ssim
print("PSNR", sum_psnr/len(images))
print("SSIM", sum_ssim/len(images))


#processAll("real_rain_img/re-1.jpg")
#processAll("real_rain_img/2-r.jpg")



