import random
import cv2
import copy
import numpy as np
import math
from skimage import io, util
from PIL import Image
import argparse

def create_gaussian_mask(n, sigma):
    gaussian_mask = [[0 for i in range(n)]for j in range(n)]
    x = np.arange(-math.floor(n/2), math.floor((n+1)/2))
    y = np.arange(-math.floor(n/2), math.floor((n+1)/2))
    a,b = np.meshgrid(x,y);
    for i in range(n):
        for j in range(n):
            p = a[i][j]
            q = b[i][j]
            c = (1/(2*3.14*sigma*sigma))
            d =- ((p*p)+(q*q))/(2*sigma*sigma)
            gaussian_mask[i][j] = c*math.exp(d)
    return gaussian_mask

def Gaussian_filtering(image_path,gaussian_mask,padding,n):
    avg = 0
    gray_image = cv2.imread(image_path,0)
    gray_scale_image=np.zeros((len(gray_image)+(2*padding),len(gray_image[0])+(2*padding)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+padding][j+padding] = gray_image[i][j]   
    gray_gaussian_image = np.zeros((len(gray_scale_image),len(gray_scale_image[0])))
    for i in range(n):
        for j in range(n):
            avg = avg+gaussian_mask[i][j]
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            s = 0
            for k in range(len(gaussian_mask)):
                for l in range(len(gaussian_mask)):
                    x = abs(k-i-padding)
                    y = abs(l-j-padding)
                    s = s+(gaussian_mask[k][l]*gray_scale_image[x][y])
            s = s/avg
            gray_gaussian_image[i][j] = s
    gray_gaussian_image = gray_gaussian_image[padding:-padding, padding:-padding]
    return gray_gaussian_image

def Salt_and_Pepper(gray_image,p):
    gray_image = cv2.imread(image_path,0)
    gray_copy = copy.deepcopy(gray_image)
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            r = random.random()
            if(r<(p/2)):
                gray_copy[i][j] = 0
            elif(r>(p/2) and r<p):
                gray_copy[i][j] = 255
            else:
                gray_copy[i][j] = gray_image[i][j]
    print("Image after salt and pepper")
    image1 = Image.fromarray((gray_copy).astype(np.uint8))
    image1.show()
    return gray_copy

def median_filtering(Noise_image,mask,padding,n):
    gray_scale_image=np.zeros((len(Noise_image)+(2*padding),len(Noise_image[0])+(2*padding)))
    for i in range(len(Noise_image)):
        for j in range(len(Noise_image[0])):
            gray_scale_image[i+padding][j+padding] = Noise_image[i][j]   
    gray_median_image=copy.deepcopy(gray_scale_image)
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            list1 = []
            for k in range(len(mask)):
                for l in range(len(mask)):
                    x = abs(k-i-1)
                    y = abs(l-j-1)
                    s = (mask[k][l]*gray_scale_image[x][y])
                    list1.append(s)
            sort = sorted(list1)
            x = sort[math.floor((n*n)/2)]
            gray_median_image[i][j] = x
    gray_median_image = gray_median_image[padding:-padding, padding:-padding]
    return gray_median_image

def mean_filtering(image_path,mask, padding):
    gray_image = cv2.imread(image_path,0)
    gray_scale_image = np.zeros((len(gray_image)+(2*padding),len(gray_image[0])+(2*padding)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+padding][j+padding] = gray_image[i][j]   
    gray_mean=copy.deepcopy(gray_scale_image)
    for i in range(padding,len(gray_scale_image)-padding):
        for j in range(padding,len(gray_scale_image[0])-padding):
            s = 0
            for k in range(len(mask)):
                for l in range(len(mask)):
                    x = abs(k-i-padding)
                    y = abs(l-j-padding)
                    s = s+(mask[k][l]*gray_scale_image[x][y])
            gray_mean[i][j] = s
    gray_mean = gray_mean[padding:-padding, padding:-padding]
    return gray_mean

def filtering(image_path, kernal_size, mode, pepper_salt, sigma, sequence=False):
    n = kernal_size
    padding = math.ceil(n/2)-1
    if mode == "mean":
        mask = np.ones((n,n),np.float32)/(n*n)
        image = mean_filtering(image_path, mask, padding)
    elif mode == "median":
        p = pepper_salt
        mask=np.ones((n,n),np.float32)
        Noise_image=Salt_and_Pepper(image_path ,p)
        image = median_filtering(Noise_image,mask, padding,n)
    elif mode == "gaussian":
        mask = create_gaussian_mask(n, sigma)
        image = Gaussian_filtering(image_path, mask, padding,n)
    res = Image.fromarray((image).astype(np.uint8))
    return res

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", required=True, type=str, help="path of image you want to quilt")
parser.add_argument("-k", "--kernal_size", type=int, default=3, help="block size in pixels")
parser.add_argument("-m", "--mode", type=str, default='mean', help="which mode -- gaussian/mean/median")
parser.add_argument("-p", "--pepper_salt", type=float, default=0.1, help="add noise to image only for median filtering")
parser.add_argument("-s", "--sigma", type=int, default=100, help="value of sigma for gaussian mask used in gaussian filtering")
args = parser.parse_args()

if __name__ == "__main__":
    image_path = args.image_path
    kernal_size = args.kernal_size
    mode = args.mode
    pepper_salt = args.pepper_salt
    sigma = args.sigma
    filtering(image_path, kernal_size, mode, pepper_salt, sigma).show()
