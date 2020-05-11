#!/usr/bin/env python
# coding: utf-8

# In[53]:


import math
import numpy as np
print("Enter the Gaussian Kernal size")
n=int(input())
x = np.arange(-math.floor(n/2), math.floor((n+1)/2))
y = np.arange(-math.floor(n/2), math.floor((n+1)/2))
a,b= np.meshgrid(x,y);
print(len(a),len(b))  


# In[54]:


gaussian_mask=[[0 for i in range(n)]for j in range(n)]
print("Enter the variance")
sigma=int(input())
for i in range(n):
    for j in range(n):
        p=a[i][j]
        q=b[i][j]
        c=(1/(2*3.14*sigma*sigma))
        d=-((p*p)+(q*q))/(2*sigma*sigma)
        gaussian_mask[i][j]=c*math.exp(d)
for i in range(n):
    print(gaussian_mask[i])


# In[55]:


import cv2
import copy
import math
import numpy as np
gray_image=cv2.imread("image_3.png",0)
#gray_copy=copy.deepcopy(gray_image)
cv2.imshow("Original Image",gray_image)
padding=math.ceil(n/2)-1
print(padding)
def Gaussian_filtering(gray_image,gaussian_mask):
    avg=0
    gray_scale_image=np.zeros((len(gray_image)+(2*padding),len(gray_image[0])+(2*padding)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+padding][j+padding]=gray_image[i][j]   
    gray_gaussian_image=np.zeros((len(gray_scale_image),len(gray_scale_image[0])))
    print((gray_scale_image.shape))
    for i in range(n):
        for j in range(n):
            avg=avg+gaussian_mask[i][j]
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            s=0
            for k in range(len(gaussian_mask)):
                for l in range(len(gaussian_mask)):
                    x=abs(k-i-padding)
                    y=abs(l-j-padding)
                    s=s+(gaussian_mask[k][l]*gray_scale_image[x][y])
            s=s/avg
            gray_gaussian_image[i][j]=s
    gray_gaussian_image=gray_gaussian_image[padding:-padding, padding:-padding]
    return gray_gaussian_image
blur_image =Gaussian_filtering(gray_image,gaussian_mask)
blur1=blur_image.astype(np.uint8)
cv2.imshow("implementation of gaussian blur",blur1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[58]:


gaussian_inbuilt=cv2.GaussianBlur(gray_image,(n,n),sigma)
cv2.imshow("inbuilt of gaussian blur",gaussian_inbuilt)
cv2.waitKey(0)
cv2.destroyAllWindows()
d=blur_image-gaussian_inbuilt
diff=d.astype(np.uint8)
cv2.imshow("Difference",diff)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




