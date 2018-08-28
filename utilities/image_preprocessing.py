import numpy as np
import cv2

def preprocessing(img, he_flag=False, clahe_flag=False, normalized_flag=False, gamma_flag=False, gamma = 1.2, **kwargs):
    if he_flag:
        img = histo_equalized(img)
    if clahe_flag:
        img = clahe_equalized(img)
    if normalized_flag:
        img = dataset_normalized(img)
    if gamma_flag:
        img = adjust_gamma(img, gamma)
    return img

def histo_equalized(img):
    img_equalized = np.empty(img.shape)
    img_equalized[0] = cv2.equalizeHist(np.array(img[0], dtype = np.uint8))
    return img_equalized

def clahe_equalized(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalized = clahe.apply(np.array(img, dtype = np.uint8))
    return img_equalized

def dataset_normalized(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255
    return img_normalized

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = cv2.LUT(np.array(img, dtype = np.uint8), table)
    return new_img
