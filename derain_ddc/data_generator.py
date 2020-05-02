#coding=utf-8
import os
import cv2
import numpy as np


dataset_folder = "/u/eot/manavm3/ORAs/data/reflection/SIR/"

#mask_truth_folder = os.path.join(dataset_folder, "map")
blend_folder = os.path.join(dataset_folder, "mixed_image_train")
gd_folder = os.path.join(dataset_folder, "ground_truth_train")
ground_truth_images_list = []

for one in os.listdir(gd_folder):
    name, ext = os.path.splitext(one)
    # path = os.path.join(gd_folder, one)
    ground_truth_images_list.append(one)


def randomCrop(image,ground_truth,crop_size):
    if image.shape!=ground_truth.shape:
        #print "if image.shape!=ground_truth.shape:"
        return cv2.resize(image, crop_size), cv2.resize(ground_truth, crop_size)
    w,h = crop_size
    w_image,h_image = image.shape[1]-1,image.shape[0]-1
    if w_image<w or h_image<h:
        #print "if w_image<w or h_image<h:"
        return cv2.resize(image,crop_size),cv2.resize(ground_truth,crop_size)
    x_range = [0,w_image-w]
    y_range = [0,h_image-h]
    x = np.random.randint(*x_range);
    y = np.random.randint(*y_range);
    return image[y:y+h,x:x+w],ground_truth[y:y+h,x:x+w]

import os

def batch_generator(batch_size):
    while True:
        X = np.zeros(shape = (batch_size,224,224,3),dtype=np.float) # 有雨的
        Y = np.zeros(shape = (batch_size,224,224,3),dtype=np.float) # gd
        Y_mask = np.zeros(shape=(batch_size, 224, 224, 1), dtype=np.float) # mask
        # Y_output = np.zeros(shape=(batch_size, 224, 224, 1), dtype=np.float) # 通过 gd 和 mask 合成雨的 output
        # Y = [Y_gd,Y_mask,X]
        #选一张图片
        # rain_image = os.path.join(blend_folder, name + ".png")
        k = 0
        for x in range(batch_size):
            # print ground_truth_images_list
            name = ground_truth_images_list[np.random.randint(0, len(ground_truth_images_list))]
            k = name[:-3].rfind('g')
            bname = name[:k] + 'm' + name[k+1:]
            blend  = os.path.join(blend_folder, bname)
            gt = os.path.join(gd_folder, name)
            #mask = os.path.join(mask_truth_folder, name + ".png")
            try:
                image_gt = cv2.imread(gt)/255.0
                #image_mask  = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)/255.0
                #image_mask = np.expand_dims(image_mask,2)
                image_blend = cv2.imread(blend)/255.0
            except:
                os.remove(blend)
                os.remove(gt)
                #os.remove(mask)
                k = 1
                break
            image_blend = cv2.resize(image_blend, (224,224), interpolation = cv2.INTER_AREA)
            image_gt = cv2.resize(image_gt, (224, 224), interpolation = cv2.INTER_AREA)
            X[x] = image_blend
            Y[x] = image_gt
        yield X, Y
#
#
if __name__ == '__main__':
#
    for x,y in batch_generator(20):
        pass





 






