import numpy as np
from imgaug import imgaug


def apply_image_aug(images, seq_det, masks=False):
    images = [np.expand_dims(np.round(image * 255.0).astype(np.uint8), axis=2) for image in images]
    if masks:
        images = seq_det.augment_images(images, hooks=get_hooks_binmasks())
    else:
        images = seq_det.augment_images(images)
    return [np.squeeze(image, axis=2) * 1.0 / 255.0 for image in images]


# change the activated augmenters for binary masks,
def activator_binmasks(images, augmenter, parents, default):
    if augmenter.name in ["GaussianBlur", "Dropout", "GaussianNoise"]:
        return False
    else:
        # default value for all other augmenters
        return default

def get_hooks_binmasks():
    return imgaug.HooksImages(activator=activator_binmasks)