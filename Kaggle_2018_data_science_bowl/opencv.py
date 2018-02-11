import cv2
import numpy as np
from matplotlib import pyplot as plt

BLACK = [255,0,0]

img1 = cv2.imread('bde3727f3a9e8b2b58f383ebc762b2157eb50cdbff23e69b025418b43967556b.png')

origin_height,origin_width=img1.shape[0],img1.shape[1]
new_size=800

#boundary: top, bottom, left, right
constant= cv2.copyMakeBorder(img1,0,new_size-origin_height,0,new_size-origin_width,cv2.BORDER_CONSTANT,value=BLACK)
cropped = constant[0:origin_height+10, 0:origin_width+10]


plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()
plt.imshow(cropped,'gray')
plt.show()
