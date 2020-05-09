import zf_unet_224_model
import data_generator as dg
from skimage.measure import compare_ssim as compare_ssim
model = zf_unet_224_model.ZF_UNET_224()
model.load_weights("temp.h5")

import numpy as np
import cv2
import os
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



image_list = []
dg.batch_generator(1)

def computeSSIM(y,ground_truth):
    y = y*255	
    y = y.astype(np.uint8)
    ground_truth = ground_truth *255
    ground_truth = ground_truth.astype(np.uint8) 
    print("psnr",psnr(y,ground_truth))
    B1,G1,R1 = cv2.split(y)
    B2, G2, R2 = cv2.split(ground_truth)
    print("ssim","B",compare_ssim(B1,B2),"G",compare_ssim(G1,G2),"R",compare_ssim(R1,R2))

import time

	
for X,Y in dg.batch_generator(12):
    t0 = time.time()
    P = model.predict(X)
    print(time.time() - t0)
    computeSSIM(P[0],Y[0])
    g = P[0]*255
    d = X[0] * 255
    image = g.astype(np.uint8)
    d= d.astype(np.uint8)
    cv2.imshow("image",image)
    cv2.imshow("k", d)
    cv2.waitKey(0)

# if __name__ == '__main__':
