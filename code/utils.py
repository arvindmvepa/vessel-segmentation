import os
import numpy as np
from PIL import Image, ImageEnhance


def get_imgs(target_dir, img_size, dataset):
    img_files, vessel_files, mask_files = None, None, None
    if dataset == 'STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)
    elif dataset == 'ARIA':
        img_files, vessel_files, mask_files = ARIA_files(target_dir)
    elif dataset == 'CHASE':
        img_files, vessel_files, mask_files = CHASE_files(target_dir)
    elif dataset == 'HRF':
        img_files, vessel_files, mask_files = HRF_files(target_dir)

    # load images
    input_imgs = imagefiles2arrs(img_files)
    vessel_imgs = imagefiles2arrs(vessel_files) // 255

    if img_size[0]!=0:
        vessel_imgs = pad_imgs(vessel_imgs, img_size)
    assert (np.min(vessel_imgs) == 0 and np.max(vessel_imgs) == 1)

    # mask_imgs = imagefiles2arrs(mask_files) // 255
    if img_size[0]!=0:
        mask_imgs = pad_imgs(mask_imgs, img_size)
    assert (np.min(mask_imgs) == 0 and np.max(mask_imgs) == 1)

    # standardize input images, 0-1
    n_all_imgs = input_imgs.shape[0]
    for index in range(n_all_imgs):
        for index2 in range(0,3):
            min = np.min(input_imgs[index, ...,index2])
            max = np.max(input_imgs[index, ...,index2])
            input_imgs[index, ...,index2] = (input_imgs[index, ...,index2] - min) / (max-min)
    if img_size[0]!=0:
        input_imgs = pad_imgs(input_imgs, img_size)
    return input_imgs, np.round(vessel_imgs), np.round(mask_imgs)


def STARE_files(data_path):
    img_dir = os.path.join(data_path, "images")
    vessel_dir = os.path.join(data_path, "vessel")
    mask_dir = os.path.join(data_path, "mask")
    img_files = all_files_under(img_dir, extension=".ppm")
    vessel_files = all_files_under(vessel_dir, extension=".ppm")
    mask_files = all_files_under(mask_dir, extension=".ppm")
    return img_files, vessel_files, mask_files


def ARIA_files(data_path):
    img_dir = os.path.join(data_path, "images")
    vessel_dir = os.path.join(data_path, "vessel")
    img_files = all_files_under(img_dir, extension=".tif")
    vessel_files = all_files_under(vessel_dir, extension=".tif")
    return img_files, vessel_files, None


def CHASE_files(data_path):
    img_dir = os.path.join(data_path, "images")
    vessel_dir = os.path.join(data_path, "vessel")
    img_files = all_files_under(img_dir, extension=".jpg")
    vessel_files = all_files_under(vessel_dir, extension=".png")
    return img_files, vessel_files, None


def HRF_files(data_path):
    img_dir = os.path.join(data_path, "images")
    vessel_dir = os.path.join(data_path, "vessel")
    mask_dir = os.path.join(data_path, "mask")
    img_files = all_files_under(img_dir, extension=".jpg")
    vessel_files = all_files_under(vessel_dir, extension=".tif")
    mask_files = all_files_under(mask_dir, extension=".tif")
    return img_files, vessel_files, mask_files


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    if sort:
        filenames = sorted(filenames)
    return filenames


def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    images_arr = None
    if len(img_shape) == 3: # RGB color images
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2: # Gray images
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)
    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)
    return images_arr


def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def pad_imgs(imgs, img_size):
    padded = None
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    target_h, target_w = img_size[0], img_size[1]
    if len(imgs.shape) == 4:
        d = imgs.shape[3]
        padded = np.zeros((imgs.shape[0], target_h, target_w, d))
    elif len(imgs.shape) == 3:
        padded = np.zeros((imgs.shape[0], target_h, target_w))
    start_h, start_w = (target_h - img_h) // 2, (target_w - img_w) // 2
    end_h, end_w = start_h + img_h, start_w + img_w
    padded[:, start_h:end_h, start_w:end_w, ...] = imgs
    return padded
