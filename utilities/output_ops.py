import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import io

from utilities.misc import find_closest_pos

# assume this is square
# TODO: return statistics on target image correlationn
def tile_images(list_of_images, new_shape = (4000, 4000), save=True, file_name = "layer1_collage.jpeg"):
    list_of_images.sort(key=lambda x: x[0])
    num_images = len(list_of_images)
    sqrt_num_images = int(num_images**(.5))
    new_image = np.zeros(new_shape)
    resize_height = int(new_shape[0] / sqrt_num_images)
    resize_width = int(new_shape[1] / sqrt_num_images)
    grid_positions = []

    for i in range(num_images):
        list_of_images[i] = cv2.resize(list_of_images[i][1], (resize_height, resize_width), cv2.INTER_AREA)

    for i in range(sqrt_num_images):
        for j in range(sqrt_num_images):
            grid_positions += [np.array((i,j))]

    for i in range(num_images):
        pos, grid_positions = find_closest_pos(grid_positions)
        x_index = resize_height*pos[0]
        y_index = resize_width*pos[1]
        new_image[x_index:x_index+resize_height, y_index:y_index+resize_width] = list_of_images[i]
    if save:
        plt.imsave(file_name, new_image)
    return new_image

def draw_results(test_inputs, test_targets, test_segmentations, test_accuracy, network, batch_num, n_examples_to_plot,
                 decision_thresh=.5):

    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10), squeeze=False)
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        axs[0][example_i].axis('off')
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[1][example_i].axis('off')
        axs[2][example_i].imshow(
            np.reshape(test_segmentations[example_i,:,:], [network.IMAGE_WIDTH, network.IMAGE_HEIGHT]),
            cmap='gray')
        axs[2][example_i].axis('off')

        test_image_thresholded = np.array(
            [0 if x < decision_thresh else 255 for x in test_segmentations[example_i,:,:].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_WIDTH, network.IMAGE_HEIGHT]),
            cmap='gray')
        axs[3][example_i].axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf