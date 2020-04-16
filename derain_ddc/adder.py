import cv2
import numpy as np

import os



image = cv2.imread("001_GT.png")
mask = cv2.imread("001_R.png",cv2.IMREAD_GRAYSCALE)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)


def addRain(origin,mask):
    imageR = (mask-128)/255.0
    image= origin.astype(np.float)/255.0

    B,G,R = cv2.split(image)
    B+=imageR
    G+=imageR
    R+=imageR
    image = cv2.merge((B,G,R))
    image *= 255
    K = image.astype(np.uint)
    return K

class DataSynthesis():

    def random_crop(self,image,size):
        x = np.random.randint(0,image.shape[1]-size[0]+1)
        y = np.random.randint(0, image.shape[0] - size[1] + 1)
        return image[y:y+size[1],x:x+size[0]]

    def __init__(self,mask_folder,attachImage_folder,output_rain,output_gt,size):
        self.bk = []
        self.mask = []
        self.output_folder = output_rain
        self.SIZE = size
        self.output_folder_gd = output_gt

        for filename in os.listdir(mask_folder):
            pathA = os.path.join(mask_folder,filename)
            self.mask.append(pathA)
        for filename in os.listdir(attachImage_folder):
            pathB = os.path.join(attachImage_folder, filename)
            self.bk.append(pathB)


    def pickOne(self,size,id):
        bk_image_path = self.bk[np.random.randint(0, len(self.bk))]

        bk_image = cv2.imread(bk_image_path)
        factor = 1.0
        bk_image = cv2.resize(bk_image,(int(224*factor),int(224*factor)))
        image_crop_bk = self.random_crop(bk_image,(224,224))
        basename,filename = os.path.split( bk_image_path)
        name,ext = os.path.splitext(filename)
        cv2.imwrite(self.output_folder_gd + "/" + str(id) + ".png", image_crop_bk)

        for x in xrange(size):
            mask_rain_path = self.mask[np.random.randint(0, len(self.mask))]
            mask_image = cv2.imread(mask_rain_path, cv2.IMREAD_GRAYSCALE)
            image_crop_rain = self.random_crop(mask_image, (224, 224))
            rain_image = addRain(image_crop_bk,image_crop_rain)
            cv2.imwrite(self.output_folder+"/"+str(id)+"_"+str(x)+".png",rain_image)




obj = DataSynthesis("/home/ly/Downloads/rain12_mask","/home/ly/Downloads/BSDS300/images/train","/home/ly/Downloads/rain_data/in","/home/ly/Downloads/rain_data/gd",(224,224))


for x in xrange(0,500):
    print x
    obj.pickOne(8,x)

