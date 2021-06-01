import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from config.logconfig import logging
import functools

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
#log.setLevel(logging.DEBUG)


@functools.lru_cache(2, typed=True)
def get_data(output_path, isTrainSet_bool):

        _ = datasets.CIFAR10(output_path, train=True, download=True)
        _ = datasets.CIFAR10(output_path, train=False, download=True)

        tensor_cifar10 = datasets.CIFAR10(output_path, train=True, download=False,
                          transform=transforms.ToTensor())
        imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
        means = imgs.view(3, -1).mean(dim=1)
        stds  = imgs.view(3, -1).std(dim=1)
        
        data = datasets.CIFAR10(
                output_path, 
                train=isTrainSet_bool, 
                download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(means,stds)
        ]))

        label_map = {0: 0, 2: 1}
        class_names = ['airplane', 'bird']
        return [(img, label_map[label])
                   for img, label in data if label in [0, 2]]


class CifarDataset(Dataset):    
    def __init__(self, output_dir, isTrainSet_bool=True):
        self.output_path = output_dir
        self.cifar2 = get_data(self.output_path, isTrainSet_bool)     

        log.info("{} {} samples".format(
            len(self.cifar2),
            "training" if isTrainSet_bool else "validation"
        ))   

    def __len__(self):
        return len(self.cifar2)

    def __getitem__(self, ndx):
        return self.cifar2[ndx]

        