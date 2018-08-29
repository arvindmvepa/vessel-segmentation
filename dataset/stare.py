from dataset.dataset_w_masks import DatasetWMasks
from network.stare import StareNetwork

class StareDataset(DatasetWMasks):

    def __init__(self, batch_size=1, WRK_DIR_PATH ="./stare", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = True,
                 masks_provided=False, init_mask_imgs=True, mask_threshold=.1, cv_train_inds = None, cv_test_inds = None):
        super(StareDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                           TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, masks_provided=masks_provided,
                                           init_mask_imgs=init_mask_imgs, mask_threshold=mask_threshold,
                                           cv_train_inds=cv_train_inds, cv_test_inds=cv_test_inds)

    @property
    def network_cls(self):
        return StareNetwork