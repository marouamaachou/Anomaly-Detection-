from typing import List, Optional, Callable

import torch
from torchvision.datasets import MNIST


class MnistAnomaly(MNIST):
    """
    A unsupervised anomaly detection version of MNIST.
    In train mode, only normal digit are available.
    In test mode, all digits are available
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        anomaly_categories: List[int] = [],
    ):
        """

        :param train: train/test set
        :param anomaly_categories: digit categories to exclude from train set
        """
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.anomaly_categories = anomaly_categories
        anomaly_cat = [True if c not in self.anomaly_categories else False for c in self.targets]
        if train == True:
            # remove digits
            self.data = self.data[anomaly_cat]
            # all digits are "normal"
            self.targets = torch.ones(len(self.data))
        else:
            # generate "abnormal","normal" categories
            self.targets = anomaly_cat
