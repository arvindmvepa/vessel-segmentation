# FCN Vessel Segmentation Network

## Setup

I used python 2.7 for development and the network only has been tested for python 2.7.

## Installation

Please install `cv2`, tensorflow 1.2, scikit-image, and statsmodels.

### Data organization

Please organize your

```
DRIVE
│
└───test
|    ├───1st_manual
|    └───2nd_manual
|    └───images
|    └───mask
│
└───training
    ├───1st_manual
    └───images
    └───mask

## Run
Before training, the 20 images of the DRIVE training datasets are pre-processed with the following transformations:
- Gray-scale conversion
- Standardization
- Contrast-limited adaptive histogram equalization (CLAHE)
- Gamma adjustment
```
We refer to the DRIVE website for the description of the data.

It is convenient to create HDF5 datasets of the ground truth, masks and images for both training and testing.
In the root folder, just run:
```
python prepare_datasets_DRIVE.py
```
The HDF5 datasets for training and testing will be created in the folder `./DRIVE_datasets_training_testing/`.
N.B: If you gave a different name for the DRIVE folder, you need to specify it in the `prepare_datasets_DRIVE.py` file.

Now we can configure the experiment. All the settings can be specified in the file `configuration.txt`, organized in the following sections:
**[data paths]**
Change these paths only if you have modified the `prepare_datasets_DRIVE.py` file.
**[experiment name]**
Choose a name for the experiment, a folder with the same name will be created and will contain all the results and the trained neural networks.
**[data attributes]**
The network is trained on sub-images (patches) of the original full images, specify here the dimension of the patches.
**[training settings]**
Here you can specify:
- *N_subimgs*: total number of patches randomly extracted from the original full images. This number must be a multiple of 20, since an equal number of patches is extracted in each of the 20 original training images.
- *inside_FOV*: choose if the patches must be selected only completely inside the FOV. The neural network correctly learns how to exclude the FOV border if also the patches including the mask are selected. However, a higher number of patches are required for training.
- *N_epochs*: number of training epochs.
- *batch_size*: mini batch size.
- *nohup*: the standard output during the training is redirected and saved in a log file.


After all the parameters have been configured, you can train the neural network with:
```
python run_training.py
```
If available, a GPU will be used.
The following files will be saved in the folder with the same name of the experiment:
- model architecture (json)
- picture of the model structure (png)
- a copy of the configuration file
- model weights at last epoch (HDF5)
- model weights at best epoch, i.e. minimum validation loss (HDF5)
- sample of the training patches and their corresponding ground truth (png)


### Evaluate the trained model
The performance of the trained model is evaluated against the DRIVE testing dataset, consisting of 20 images (as many as in the training set).

The parameters for the testing can be tuned again in the `configuration.txt` file, specifically in the [testing settings] section, as described below:
**[testing settings]**
- *best_last*: choose the model for prediction on the testing dataset: best = the model with the lowest validation loss obtained during the training; last = the model at the last epoch.
- *full_images_to_test*: number of full images for testing, max 20.
- *N_group_visual*: choose how many images per row in the saved figures.
- *average_mode*: if true, the predicted vessel probability for each pixel is computed by averaging the predicted probability over multiple overlapping patches covering the same pixel.
- *stride_height*: relevant only if average_mode is True. The stride along the height for the overlapping patches, smaller stride gives higher number of patches.
- *stride_width*: same as stride_height.
- *nohup*: the standard output during the prediction is redirected and saved in a log file.

The section **[experiment name]** must be the name of the experiment you want to test, while **[data paths]** contains the paths to the testing datasets. Now the section **[training settings]** will be ignored.

Run testing by:
```
python run_testing.py
```
If available, a GPU will be used.
The following files will be saved in the folder with same name of the experiment:
- The ROC curve  (png)
- The Precision-recall curve (png)
- Picture of all the testing pre-processed images (png)
- Picture of all the corresponding segmentation ground truth (png)
- Picture of all the corresponding segmentation predictions (png)
- One or more pictures including (top to bottom): original pre-processed image, ground truth, prediction
- Report on the performance

All the results are referred only to the pixels belonging to the FOV, selected by the masks included in the DRIVE database