import os
import cv2
import numpy as np


dataset_folder = "/home/ly/workspace/rainy_image_dataset"

ground_truth_folder = os.path.join(dataset_folder, "ground truth")
rainy_image_folder = os.path.join(dataset_folder, "rainy image")
ground_truth_images_list = []

for one in os.listdir(ground_truth_folder):
    name, ext = os.path.splitext(one)
    path = os.path.join(ground_truth_folder, one)
    ground_truth_images_list.append([path, name])


def randomCrop(image,ground_truth,crop_size):
    if image.shape!=ground_truth.shape:
        print "if image.shape!=ground_truth.shape:"
        return cv2.resize(image, crop_size), cv2.resize(ground_truth, crop_size)
    w,h = crop_size
    w_image,h_image = image.shape[1]-1,image.shape[0]-1
    if w_image<w or h_image<h:
        return cv2.resize(image,crop_size),cv2.resize(ground_truth,crop_size)
    x_range = [0,w_image-w]
    y_range = [0,h_image-h]
    x = np.random.randint(*x_range);
    y = np.random.randint(*y_range);
    return image[y:y+h,x:x+w],ground_truth[y:y+h,x:x+w]



def batch_generator(batch_size):
    while True:
        X = np.zeros(shape = (batch_size,224,224,3),dtype=np.float)
        Y = np.zeros(shape = (batch_size,224,224,3),dtype=np.float)
        path_gt, name = ground_truth_images_list[np.random.randint(0, len(ground_truth_images_list))]
        rain_image = os.path.join(rainy_image_folder, name + "_" + str(np.random.randint(1, 14)) + ".jpg")
        A = cv2.imread(path_gt)
        B = cv2.imread(rain_image)
        scale_factor = 0.8+np.random.random()
        h,w,c = A.shape
        h,w = int(scale_factor*h),int(scale_factor*w)
        A = cv2.resize(A,(w,h))
        B = cv2.resize(B, (w, h))
        for x in xrange(batch_size):
            # print ground_truth_images_list
            if x%10==0:
                path_gt, name = ground_truth_images_list[np.random.randint(0, len(ground_truth_images_list))]
                rain_image = os.path.join(rainy_image_folder, name + "_" + str(np.random.randint(1,15)) + ".jpg")
                A = cv2.imread(path_gt)
                B = cv2.imread(rain_image)

            output,input = randomCrop(A,B,(224,224))
            # if x%100==0:
            #  cv2.imshow("input",input)
            # cv2.imshow("output",output)
            # cv2.waitKey(0)

            input_image = input.astype(np.float)/255
            output_image = output.astype(np.float)/255
            X[x] = input_image
            Y[x] = output_image
        yield X,Y
#
#
# if __name__ == '__main__':
#
# for x,y in batch_generator(20):
#     print x,y
#
#











