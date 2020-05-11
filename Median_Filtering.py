#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import math
import copy
import numpy as np
import random
gray_image =cv2.imread("image_2.png",0)
print(gray_image.shape)
gray_copy=copy.deepcopy(gray_image)
def Salt_and_Pepper(gray_image,p):
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            r=random.random()
            if(r<(p/2)):
                gray_copy[i][j]=0
            elif(r>(p/2) and r<p):
                gray_copy[i][j]=255
            else:
                gray_copy[i][j]=gray_image[i][j]
    return gray_copy
print("Enter the amount of noise you want to insert")
p=float(input())
Noise_image=Salt_and_Pepper(gray_image,p)
cv2.imshow("implement image after salt and pepper",Noise_image)
#cv2.imwrite("/noise2",Noise_image)
n=int(input())
mask=np.ones((n,n),np.float32)
padding=math.ceil(n/2)-1

def median_filtering(Noise_image,mask):
    gray_scale_image=np.zeros((len(Noise_image)+(2*padding),len(Noise_image[0])+(2*padding)))
    for i in range(len(Noise_image)):
        for j in range(len(Noise_image[0])):
            gray_scale_image[i+padding][j+padding]=Noise_image[i][j]   
    gray_median_image=copy.deepcopy(gray_scale_image)
    print((gray_scale_image.shape))
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            list1=[]
            for k in range(len(mask)):
                for l in range(len(mask)):
                    x=abs(k-i-1)
                    y=abs(l-j-1)
                    s=(mask[k][l]*gray_scale_image[x][y])
                    list1.append(s)
            sort=sorted(list1)
            x=sort[math.floor((n*n)/2)]
            gray_median_image[i][j]=x
    gray_median_image=gray_median_image[padding:-padding, padding:-padding]
    print(gray_median_image.shape)
    return gray_median_image
blur_image=median_filtering(Noise_image,mask)
blur_image=blur_image.astype(np.uint8)
cv2.imshow("implementation of median filtering",blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


inbuilt_blur=cv2.medianBlur(Noise_image,n)
inbuilt_blur=inbuilt_blur.astype(np.uint8)
cv2.imshow("inbuilt of median filtering",inbuilt_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


diff=blur_image-inbuilt_blur
diff=diff.astype(np.uint8)
cv2.imshow("difference of median filtering",diff)
cv2.waitKey(0)
cv2.destroyAllWindows()   


# In[ ]:




