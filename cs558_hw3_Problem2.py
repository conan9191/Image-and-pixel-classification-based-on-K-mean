from PIL import Image
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def loadImage(labelName):
    imgname = "sky/" + labelName + ".jpg"
    pre_image = cv2.imread(imgname)
    print(labelName,pre_image)
    print(pre_image.shape)
    return pre_image

def generateTrainingData(trainImag, mask):
    row, col ,cha = trainImag.shape
    sky = []
    non_sky = []
    for i in range(row):
        for j in range(col):
            if mask[i][j][0] == 255 and mask[i][j][1] == 255 and mask[i][j][2] == 255:
                sky.append(trainImag[i][j])
            else:
                non_sky.append(trainImag[i][j])
    return sky, non_sky

def generateVisualWords(sky, non_sky):
    sky_10 = KMeans(n_clusters=10)
    non_sky_10 = KMeans(n_clusters=10)
    sky_10.fit(sky)
    non_sky_10.fit(non_sky)
    centroids_sky = sky_10.cluster_centers_
    centroids_non_sky = non_sky_10.cluster_centers_
    sky_label = np.ones((10, 1))
    non_sky_label = np.zeros((10, 1))
    words = np.vstack((centroids_sky, centroids_non_sky))
    labels = np.vstack((sky_label, non_sky_label))
    return words, labels


if __name__ == "__main__":
    '''Load image'''
    train_img = loadImage("sky_train")
    mask = loadImage("sky_train_gimp")
    test_img = []
    for i in range(4):
        num = str(i + 1).zfill(1)
        test_img.append(loadImage("sky_test"+num))

    '''genetate training data '''
    sky, non_sky = generateTrainingData(train_img, mask)

    '''generate visual words'''
    words, labels = generateVisualWords(sky, non_sky)

    '''find the nearest word'''
    for t in range (len(test_img)):
        num = str(t + 1).zfill(1)
        test = test_img[t]
        cv2.imshow("test_" + num, test)
        row, col, cha = test.shape
        for i in range(row):
            for j in range(col):
                minIndex = -1
                minDist = np.inf
                for w in range(len(words)):
                    sub = test[i][j][:] - words[w]
                    dis = np.sqrt(sub[0] ** 2 + sub[1] ** 2 + sub[2] ** 2)
                    if dis < minDist:
                        minDist = dis
                        minIndex = w
                '''Generate an output image '''
                if labels[minIndex] == 1:
                    test[i][j][0] = 255
                    test[i][j][1] = 0
                    test[i][j][2] = 0
        cv2.imshow("output_"+num, test)
    cv2.waitKey()
    cv2.destroyAllWindows()
