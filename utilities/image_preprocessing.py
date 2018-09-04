import numpy as np
import cv2

def preprocessing(img, histo_eq=False, clahe_kwargs=None, per_image_normalization=False, gamma=None, **kwargs):
    print("pre-processing options: hist_eq {}, clahe_kwargs {}, per_image_normalization {}, gamma{}".format(histo_eq,
                                                                                                            clahe_kwargs,
                                                                                                            per_image_normalization,
                                                                                                            gamma))
    if histo_eq:
        print("histo eq")
        img = histo_equalized(img)
    if clahe_kwargs:
        print("clahe")
        img = clahe_equalized(img, **clahe_kwargs)
    if per_image_normalization:
        print("per image normalization")
        img = per_image_normalize(img)
    if gamma:
        print("gamma")
        img = adjust_gamma(img, gamma)
    return img

def histo_equalized(img):
    img_equalized = cv2.equalizeHist(np.array(img, dtype = np.uint8))
    return img_equalized

def clahe_equalized(img, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_equalized = clahe.apply(np.array(img, dtype = np.uint8))
    return img_equalized

def per_image_normalize(img):
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
