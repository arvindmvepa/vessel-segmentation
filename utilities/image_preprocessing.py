import numpy as np
import cv2

def preprocessing(img, histo_eq=False, clahe_kwargs=None, gamma=None, per_image_z_score_norm=False,
                  per_image_zero_center=False, per_image_zero_center_scale=None, **kwargs):
    if histo_eq:
        img = histo_equalized(img)
    if clahe_kwargs:
        img = clahe_equalized(img, **clahe_kwargs)
    if gamma:
        img = adjust_gamma(img, gamma)
    if per_image_z_score_norm:
        img = per_image_z_score_norm(img)
    if per_image_zero_center:
        img = per_image_zero_center(img)
    if per_image_zero_center_scale:
        img = per_image_zero_center_scale(img)
    return img

def histo_equalized(img):
    img_equalized = cv2.equalizeHist(np.array(img, dtype = np.uint8))
    return img_equalized

def clahe_equalized(img, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_equalized = clahe.apply(np.array(img, dtype = np.uint8))
    return img_equalized

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = cv2.LUT(np.array(img, dtype = np.uint8), table)
    return new_img

def per_image_z_score_norm(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std
    return img_normalized

def per_image_zero_center(img):
    img_mean = np.mean(img)
    return img - img_mean

def per_image_zero_center_scale(img):
    img_mean = np.mean(img)
    img = img - img_mean
    min_val = np.min(img)
    max_val = np.max(img)
    return 2*(img - min_val)/(max_val-min_val)-1

def apply_normalization(imgs, zero_center=False, zero_center_scale=False, z_score_norm=False, train_params=None):
    """If training, calculate results on train data. otherwise use test data"""
    print("apply normalization condition statement")
    if zero_center or zero_center_scale:
        print("apply zero_center")
        # zero center by train mean and scale by [-1,1]
        if not train_params:
            print("zc train: re-calculate mean")
            mu = np.mean(imgs)
        else:
            print("zc test: use train params")
            mu = train_params[0]
        zero_centered = imgs - mu
        if zero_center_scale:
            print("apply zero_center scale")
            if not train_params:
                print("zcs train: re-calculate values")
                min_val = np.min(zero_centered,axis=(0,1,2))
                max_val = np.max(zero_centered,axis=(0,1,2))
                return 2*(zero_centered - min_val)/(max_val-min_val)-1, (mu, min_val, max_val)
            else:
                print("zcs test: use train params")
                return 2 * (zero_centered - train_params[1]) / (train_params[2] - train_params[1]) - 1, \
                       (mu, train_params[1], train_params[2])
        else:
            print("don't apply zero_center scale")
            return zero_centered, (mu,)
    elif z_score_norm:
        print("apply z score norm")
        # normalize by z-score from train data
        if not train_params:
            print("zcn train: re-calculate values")
            mu, std = np.mean(imgs), np.std(imgs)
        else:
            print("zcn test: use train params")
            mu, std = train_params
        return (imgs - mu)/std, (mu, std)
    else:
        print("no normalization applied")
        return imgs, None
