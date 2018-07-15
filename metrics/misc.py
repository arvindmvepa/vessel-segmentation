import tensorflow as tf
from utilities.mask_ops import mask_mean_diff

# currently not working
def dice_coe(output, target, mask=None, num_batches=1, loss_type='jaccard', axis=None, smooth=1e-5):
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    if mask != None:
        dice = mask_mean_diff(dice, mask, num_batches)
    else:
        dice = tf.reduce_mean(dice)
    return dice

# currently not working
def dice_hard_coe(output, target, mask=None, num_batches=1, threshold=0.5, axis=None, smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    if mask != None:
        hard_dice = mask_mean_diff(hard_dice, mask, num_batches)
    else:
        hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

# currently not working
def iou_coe(output, target, mask=None, num_batches=1, threshold=0.5, axis=None, smooth=1e-5):
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)
    batch_iou = (inse + smooth) / (union + smooth)
    if mask != None:
        iou = mask_mean_diff(batch_iou, mask, num_batches)
    else:
        iou = tf.reduce_mean(batch_iou)
    return iou
