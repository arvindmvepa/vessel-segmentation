"""This is the file for the DRIVE network subclass"""

from network.retinal_w_masks import RetinalWMasksNetwork

class DriveNetwork(RetinalWMasksNetwork):

    # actual image dimensions 
    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584


    
    
    def __init__(self, weight_init,learningrate,Beta1,Beta2,epsilon,keep_prob,Layer_param,regularizer=None,Relu=False,layers=None, skip_connections=True,**kwargs):
        super(DriveNetwork, self).__init__(weight_init,learningrate,Beta1,Beta2,epsilon,keep_prob,Layer_param,layers=layers,skip_connections=skip_connections, **kwargs)
