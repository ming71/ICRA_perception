import cv2 
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('data/raw.png')
h, w, c = img.shape
pts1 = np.float32([[492,344],[959,329],[523,767],[117,791]])
pts2 = np.float32([[8080,0],[8080,4480],[1630,3545],[1500,2140]]) 

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(8080, 4480))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()