from PIL import Image
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def loadImage(labelName,bin):
    imgname = "ImClass/" + labelName + ".jpg"
    pre_image = cv2.imread(imgname)
    row, col, cha = pre_image.shape
    b, g, r = cv2.split(pre_image)
    hist_b = np.zeros([1, bin])
    hist_g = np.zeros([1, bin])
    hist_r = np.zeros([1, bin])
    for i in range(row):
        for j in range(col):
            ind_b = int(np.floor(b[i][j] / (256 / bin)))
            ind_g = int(np.floor(g[i][j] / (256 / bin)))
            ind_r = int(np.floor(r[i][j] / (256 / bin)))
            hist_b[0, ind_b] = hist_b[0, ind_b] + 1
            hist_g[0, ind_g] = hist_g[0, ind_g] + 1
            hist_r[0, ind_r] = hist_r[0, ind_r] + 1

    # hist_bb = cv2.calcHist([pre_image],[0],None,[8],[0,255])
    # hist_gg = cv2.calcHist([pre_image],[1],None,[8],[0,255])
    # hist_rr = cv2.calcHist([pre_image],[2],None,[8],[0,255])
    hist = np.hstack((hist_b,hist_g,hist_r))
    # print(hist)
    # print(hist.shape)

    return hist

def k_nearest_neighbor(training_X, training_y, test_X, test_y, k):
    row, col = test_X.shape
    predicts = []
    for i in range(row):
        dis = np.sqrt(np.sum((test_X[i,:] - training_X) ** 2, axis=1))
        ind = dis.argsort()[0: k]
        candidate = training_y[ind].reshape((-1,),order='F')
        vote = np.bincount(candidate)
        if vote.size == 4 and vote[1]==vote[2] and vote[2]==vote[3]:
            predicts.append(candidate[0])
        else:
            predicts.append(vote.argmax())
        print("Test i􏰐mage",i ,"o􏰑f cl􏰒ass ",test_y[i]," has been􏰓 assig􏰓ned t􏰑o cl􏰒ass ",predicts[i])
        # print("dis",i,":",dis)
        # print("sort", i, ":", ind)
        # print("candidate:",candidate)
        # print("vote",vote)
    print(predicts)
    return predicts


if __name__ == "__main__":

    '''using 4, 8, 16, 32 bins'''
    for b in range(2,6):
        bb = 2 ** b
        traning_num = 12
        test_num = 12
        bin = bb
        training_X = np.zeros([traning_num, bin * 3])
        training_y = np.zeros([traning_num, 1],dtype=np.int)
        test_X = np.zeros([test_num, bin * 3])
        test_y = np.zeros([test_num, 1],dtype=np.int)
        iteration = 0

        ''' training data'''
        for i in range(int(traning_num / 3)):
            num = str(i+1).zfill(1)
            filename1 = "coast_train" + num
            filename2 = "forest_train" + num
            filename3 = "insidecity_train" + num
            hist1 = loadImage(filename1, bin)
            training_X[iteration,:] = hist1
            training_y[iteration,:] = 1
            iteration += 1
            hist2 = loadImage(filename2, bin)
            training_X[iteration, :] = hist2
            training_y[iteration, :] = 2
            iteration += 1
            hist3 = loadImage(filename3, bin)
            training_X[iteration, :] = hist3
            training_y[iteration, :] = 3
            iteration += 1

        ''' test data'''
        iteration2 = 0
        for i2 in range(int(test_num / 3)):
            num = str(i2+1).zfill(1)
            filename1 = "coast_test" + num
            filename2 = "forest_test" + num
            filename3 = "insidecity_test" + num
            hist1 = loadImage(filename1, bin)
            test_X[iteration2, :] = hist1
            test_y[iteration2, :] = 1
            iteration2 += 1
            hist2 = loadImage(filename2, bin)
            test_X[iteration2, :] = hist2
            test_y[iteration2, :] = 2
            iteration2 += 1
            hist3 = loadImage(filename3, bin)
            test_X[iteration2, :] = hist3
            test_y[iteration2, :] = 3
            iteration2 += 1

        ''' do 1-nearest neighbor '''
        predict1 = k_nearest_neighbor(training_X, training_y, test_X, test_y, 1)
        error = 0
        for i in range(test_num):
            if test_y[i, 0] != predict1[i]:
                error += 1
        accuracy = (test_num - error) / test_num
        print("Accurate(",bb ," bins with 1-nearest neighbor): ", np.round(accuracy,2))
        print("---------------------------")

        if bin == 8:
            ''' do 3-nearest neighbor '''
            predict3 = k_nearest_neighbor(training_X, training_y, test_X, test_y, 3)
            error = 0
            for i in range(test_num):
                if test_y[i, 0] != predict3[i]:
                    error += 1
            accuracy = (test_num - error) / test_num
            print("Accurate(",bb ," bins with 3-nearest neighbor): ", np.round(accuracy, 2))
            print("---------------------------")




