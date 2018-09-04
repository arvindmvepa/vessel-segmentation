from dataset.dataset_w_masks import DatasetWMasks
from network.stare import StareNetwork

class StareDataset(DatasetWMasks):

    def __init__(self, **kwargs):
        super(StareDataset, self).__init__( **kwargs)

    @property
    def network_cls(self):
        return StareNetwork