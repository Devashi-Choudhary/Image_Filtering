#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import copy
import math
import numpy as np
gray_image=cv2.imread("image_1.jpg",0)
print("Enter the kernal size")
n=int(input())
mask=np.ones((n,n),np.float32)/(n*n)
gray_copy=copy.deepcopy(gray_image)
padding=math.ceil(n/2)-1
def mean_filtering(gray_image,mask):
    gray_scale_image=np.zeros((len(gray_image)+(2*padding),len(gray_image[0])+(2*padding)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+padding][j+padding]=gray_image[i][j]   
    gray_mean=copy.deepcopy(gray_scale_image)
    print((gray_scale_image.shape))
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            s=0
            for k in range(len(mask)):
                for l in range(len(mask)):
                    x=abs(k-i-padding)
                    y=abs(l-j-padding)
                    s=s+(mask[k][l]*gray_scale_image[x][y])
            gray_mean[i][j]=s
    gray_mean=gray_mean[padding:-padding, padding:-padding]
    print(gray_mean.shape)
    return gray_mean
blur_image =mean_filtering(gray_image,mask)
blur_image=blur_image.astype(np.uint8)
cv2.imshow("Implementation of Mean Filtering",blur_image)
#cv2.imwrite("Implementation of Mean Filtering",blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


inbuilt_blur=cv2.filter2D(gray_copy,-1,mask)
#cv2.imshow("Implementation of mean filtering",blur_image)
cv2.imshow("inbuilt image of mean filtering",inbuilt_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


diff=inbuilt_blur-blur_image
difference=diff.astype(np.uint8)
#diff=diff.astype(np.uint8)
cv2.imshow("Comparison of Implementation and Inbuilt of mean filtering",difference)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




