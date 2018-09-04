from dataset.dataset_w_masks import DatasetWMasks
from network.chase import ChaseNetwork

class ChaseDataset(DatasetWMasks):

    def __init__(self, **kwargs):
        super(ChaseDataset, self).__init__(**kwargs)

    @property
    def network_cls(self):
        return ChaseNetwork
