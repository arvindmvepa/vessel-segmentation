import utils as utils
from PIL import Image


#read images
class Dataset():
    def __init__(self, dataset, testflag, maxsize=0):#testflag is a bool, meaning test or train; if have maxsize, means padding
        self.dataset = dataset
        self.testflag = testflag
        self.image_size = (maxsize,maxsize)
        self.train_dir = "../data/{}/training/".format(self.dataset)
        self.test_dir = "../data/{}/test/".format(self.dataset)
        self.num_train, self.num_test = 0, 0
        self.test_imgs, self.test_vessels, self.test_masks, self.train_imgs, self.train_vessels, self.train_masks = None, None, None, None, None, None

        if self.dataset == 'STARE':
            self.image_shape = (605,700)
        elif self.dataset == 'ARIA':
            self.image_shape = (576, 768)
        elif self.dataset == 'CHASE':
            self.image_shape = (960, 999)
        elif self.dataset == 'HRF':
            self.image_shape = (2336, 3504)

        self._read_data() # read training or test data


    def _read_data(self):
        # read test images and vessels in the memory
        if self.testflag == False:
            self.test_imgs, self.test_vessels, self.test_masks = utils.get_imgs(
                target_dir=self.test_dir, img_size=self.image_size, dataset=self.dataset)
            self.num_test = self.test_imgs.shape[0]
            #print ('self.test_imgs.shape[0]=',self.test_imgs.shape[0])
            #print ('self.test_imgs.shape[1]=',self.test_imgs.shape[1])
            #print ('self.test_imgs.shape[2]=',self.test_imgs.shape[2])
            #print ('self.test_imgs.shape[3]=',self.test_imgs.shape[3])
        # read training images and vessels in the memory
        else:
            self.train_imgs, self.train_vessels, self.train_masks = utils.get_imgs(
                target_dir=self.train_dir, img_size=self.image_size,dataset=self.dataset)
            self.num_train = self.train_imgs.shape[0]
            #print ('self.train_imgs.shape[0]=',self.train_imgs.shape[0])
            #print ('self.train_imgs.shape[1]=',self.train_imgs.shape[1])
            #print ('self.train_imgs.shape[2]=',self.train_imgs.shape[2])
            #print ('self.train_imgs.shape[3]=',self.train_imgs.shape[3])
