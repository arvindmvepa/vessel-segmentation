# FCN Vessel Segmentation Network

## Setup

Python 2.7 was used for development and the network only has been tested for python 2.7.

## Installation

Install `cv2`, tensorflow 1.2, imgaug, scikit-image, and statsmodels.

### Data organization

Organize the data in your drive directory as follows:

```
drive
│
└───test
|    ├───images
|    └───masks
|    └───targets (1st manual)
|
│
└───train
    ├───images
    └───masks
    └───targets
```
Organize the data in your dsa directory as follows:

```
dsa
│
└───test
|    ├───images
|    └───targets1
|    └───targets2
|
│
└───train
    ├───images
    └───targets1
    └───targets2
```
When initializing an instance of the `Job` class, pass the keyword argument `OUTPUTS_DIR_PATH` with the location on your
workspace where you want to store your experiment results. Also, when using one of the run methods in the `Job` class,
pass the keyword argument `WRK_DIR_PATH` to specify the location in which the above drive or dsa directory is located.

## Run
Run the `main.py` file to run one of the pre-defined configurations. The code can be changed as needed for needed
purposes.