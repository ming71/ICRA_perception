import os 
import cv2 
import glob
import imutils
import numpy as np
from matplotlib import pyplot as plt

def template_matching(img):
    tpls = glob.glob('data/template/*.*') 
    criteria = cv2.TM_CCOEFF_NORMED 
    scores = []
    names = []
    for tpl in tpls:
        tpl_name = os.path.split(tpl)[1].strip('.png')
        img_t = cv2.imread(tpl)
        img_t = cv2.resize(img_t, (200,200))
        th, tw = img_t.shape[:2]  
        score = cv2.matchTemplate(img, img_t, criteria)   #像素点的相关度量值
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) 
        # tl = min_loc if criteria == cv2.TM_SQDIFF_NORMED else max_loc
        # br = (tl[0]+tw, tl[1]+th)
        scores.append(score[0].item())
        names.append(tpl_name)
        # cv2.rectangle(img, tl, br, (0, 0, 255), 2)
        # cv2.imshow("match-" + np.str(md), result)
    idx = scores.index(max(scores))
    if scores[idx] > 0.5:
        return names[idx]
    else:
        return 'False'



def masked_color(img, color='blue'):
    lower_blue =  np.array([100,43,46])
    higher_blue = np.array([124,255,255])
    lower_red1 =  np.array([156, 43, 46])
    higher_red1 = np.array([180, 255, 255])
    lower_red2 =  np.array([0, 43, 46])
    higher_red2 = np.array([10, 255, 255])

    # bgr2hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color mask
    if color == 'blue':
        mask = cv2.inRange(hsv, lower_blue, higher_blue) 
        left = cv2.bitwise_and(img,img,mask=mask)
    elif color =='red':
        mask = cv2.inRange(hsv, lower_red1, higher_red1) + cv2.inRange(hsv, lower_red2, higher_red2)
        left = cv2.bitwise_and(img,img,mask=mask)
    else:
        raise NotImplementedError
    return left


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    # gray = cv2.bilateralFilter(gray, 3, 15, 15)
    # edge = cv2.Canny(binary,10, 200) 
    # cv2.imshow('res', binary)
    # cv2.waitKey(0)

    return binary

def find_contours(edge):
    contours = cv2.findContours(edge.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse=True)[:20]
    screenCnts = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            screenCnts.append(approx)

    if len(screenCnts) == 0:
        detected = 0
        print ("No contour detected")
    elif len(screenCnts) == 0:
        screenCnts = [screenCnts]
    else:
        detected = 1
    return screenCnts, detected


def PerspectiveTransform(img, pts1, pts2):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(200, 200))
    return dst

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    return np.array([tl, tr, br, bl], dtype="float32")

if __name__ == "__main__":

    img = cv2.imread('data/imgs/8.jpg')
    img = cv2.resize(img,(768,768))

    leftb = masked_color(img, color='blue')
    leftr = masked_color(img, color='red')
    left = leftb + leftr

    edge = edge_detection(left)

    screenCnts, flag = find_contours(edge)
    if flag != 0:
        dets = []
        for screenCnt in screenCnts:
            polygon = screenCnt.reshape(1,4,2).squeeze().astype('float32')
            cropped = PerspectiveTransform(img, order_points(polygon), np.float32([[0,0],[200,0],[200,200],[0,200]]))
            # cv2.drawContours(img, polygon[:,np.newaxis,:].astype('int32'), -1, (0, 0, 255), 3)
            res = template_matching(cropped)
            if (res != 'False') and (res not in dets):
                print(res)
                cv2.imshow(res, cropped)
            dets.append(res)
        cv2.waitKey(0)




