import tensorflow as tf

def mask_op_and_mask_mean(correct_pred, mask, num_batches=1, height=584, width=565):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, height, width)


def mask_op_and_mask_mean_diff(correct_pred, mask, num_batches=1, height=584, width=565):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, height,  width)


def mask_mean(masked_pred, mask, num_batches=1, height=584, width=565):
    ones = tf.ones([num_batches, height, width, 1], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype=tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)


def mask_mean_diff(masked_pred, mask, num_batches=1, height=584, width=565):
    ones = tf.ones([num_batches, height, width], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype=tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)