# Image Filters
Image Filtering is a technique for modifying or enhancing an image. You can filter an image to emphasize certain features or remove other features. 
Image processing operations implemented with filtering include smoothing, sharpening, and edge enhancement. The goal of this project is to implement
mean, median and gaussian filters from scratch.

# How to execute code:
1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. You can open the folder and run Image_Filtering.py on command prompt.
> `python Image_Filtering.py --i <image_path> --k <int> --m <str gaussian/mean/median> --p <only for median filter> -s <only for gaussian filter>`

where --i is path to image, --k is kernal size, --m is mode, --p is pepper and salt noise(only works in case median filter), --s is sigma for calculating gaussian mask.
for example
1. for mean filter : `python Image_Filtering.py --i data/image_1.jpg --k 5 --m mean` default value --m is mean and --k is 3.
2. for median filtering : `python Image_Filtering.py --i data/image_1.jpg --k 5 --m median --p 0.2` default value p is 0.1 and --k is 3.
3. for median filtering : `python Image_Filtering.py --i data/image_1.jpg --k 5 --m gaussian --s 150` default value s is 100 and --k is 3.

**Note :** For more details about implementation, result, and analysis, go through the `Image_Filtering.pdf` file.
