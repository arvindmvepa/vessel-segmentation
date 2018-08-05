import tensorflow as tf

def mask_op_and_mask_mean(correct_pred, mask, num_batches=1, width=565, height=584):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, width, height)


def mask_op_and_mask_mean_diff(correct_pred, mask, num_batches=1, width=565, height=584):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, width, height)


def mask_mean(masked_pred, mask, num_batches=1, width=565, height=584):
    ones = tf.ones([num_batches, width, height, 1], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype=tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)


def mask_mean_diff(masked_pred, mask, num_batches=1, width=565, height=584):
    ones = tf.ones([num_batches, width, height], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype=tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)