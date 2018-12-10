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
        print("per image z score norm")
        img = apply_per_image_z_score_norm(img)
    if per_image_zero_center:
        print("per image zero center")
        img = apply_per_image_zero_center(img)
    if per_image_zero_center_scale:
        print("per image zero center scale")
        img = apply_per_image_zero_center_scale(img)
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

def apply_per_image_z_score_norm(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    print("per image z score norm mean {} std {}".format(img_mean, img_std))
    img_normalized = (img - img_mean) / img_std
    return img_normalized

def apply_per_image_zero_center(img):
    img_mean = np.mean(img)
    print("per image zero center mean {}".format(img_mean))
    return img - img_mean

def apply_per_image_zero_center_scale(img):
    img_mean = np.mean(img)
    img = img - img_mean
    min_val = np.min(img)
    max_val = np.max(img)
    print("per image zero center scale mean {} min {} max {}".format(img_mean, min_val, max_val))
    return 2*(img - min_val)/(max_val-min_val)-1

def apply_dataset_normalization(imgs, zero_center=False, zero_center_scale=False, z_score_norm=False, train_params=None):
    """If training, calculate results on train data. otherwise use test data"""
    print("apply normalization condition statement")
    if zero_center or zero_center_scale:
        print("apply zero_center")
        # zero center by train mean and scale by [-1,1]
        if not train_params:
            mu = np.mean(imgs)
            print("zc train: re-calculate mean {}".format(mu))
        else:
            mu = train_params[0]
            print("zc test: use train params mu {}".format(mu))
        zero_centered = imgs - mu
        if zero_center_scale:
            print("apply zero_center scale")
            if not train_params:
                min_val = np.min(zero_centered)
                max_val = np.max(zero_centered)
                print("zcs train: re-calculate min {} max {}".format(min_val, max_val))
            else:
                min_val = train_params[1]
                max_val = train_params[2]
                print("zcs test: use train params min {} max {}".format(min_val, max_val))
            return 2*(zero_centered-min_val)/(max_val-min_val)-1, (mu, min_val, max_val)
        else:
            print("don't apply zero_center scale")
            return zero_centered, (mu,)
    elif z_score_norm:
        print("apply z score norm")
        # normalize by z-score from train data
        if not train_params:
            mu, std = np.mean(imgs), np.std(imgs)
            print("zcn train: re-calculate values mean {} std {}".format(mu,std))
        else:
            mu, std = train_params
            print("zcn test: use train params mean {} std {}".format(mu,std))
        return (imgs - mu)/std, (mu, std)
    else:
        print("no normalization applied")
        return imgs, None
