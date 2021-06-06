import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from config.logconfig import logging
import functools
import random, math

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
#log.setLevel(logging.DEBUG)


@functools.lru_cache(1, typed=True)
def get_data(output_path):
        _ = datasets.CIFAR10(output_path, train=True, download=True)
        _ = datasets.CIFAR10(output_path, train=False, download=True)
        tensor_cifar10 = datasets.CIFAR10(output_path, train=True, download=False,
                          transform=transforms.ToTensor())
                          
        imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
        means = imgs.view(3, -1).mean(dim=1)
        stds  = imgs.view(3, -1).std(dim=1)
        return means, stds

class getAugmentedTensor():
    def __init__(self, augmentation_dict):

        self.augmentation_dict = augmentation_dict
        self.transform_t = torch.eye(3)
        
        for i in range(2):
            if 'flip' in self.augmentation_dict:
                if random.random() > 0.5:
                    self.transform_t[i,i] *= -1

            if 'offset' in self.augmentation_dict:
                offset_float = self.augmentation_dict['offset']
                random_float = (random.random() * 2 - 1)
                self.transform_t[i,2] = offset_float * random_float

            if 'scale' in self.augmentation_dict:
                scale_float = self.augmentation_dict['scale']
                random_float = (random.random() * 2 - 1)
                self.transform_t[i,i] *= 1.0 + scale_float * random_float


        if 'rotate' in self.augmentation_dict:
            angle_rad = (2*random.random()-1) * math.pi * 2 * self.augmentation_dict['rotate']
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

            self.transform_t @= rotation_t

        
    def __call__(self,tensor):
        tensor_ = tensor.unsqueeze(0)

        affine_t = F.affine_grid(
                self.transform_t[:2].unsqueeze(0).to(torch.float32),
                tensor_.size(),
                align_corners=False
            )

        augmented_chunk = F.grid_sample(
                tensor_,
                affine_t,
                padding_mode='border',
                align_corners=False,
            ).to('cpu')

        if 'noise' in self.augmentation_dict:
            noise_t = torch.randn_like(augmented_chunk)
            noise_t *= self.augmentation_dict['noise']

            augmented_chunk += noise_t

        return augmented_chunk[0]



class CifarDataset(Dataset):    
    def __init__(self, 
                output_path, 
                isTrainSet_bool=True,
                augmentation_dict=None):

        self.output_path = output_path
        self.augmentation_dict = augmentation_dict

        self.means, self.stds = get_data(self.output_path)
        
        if self.augmentation_dict:
            transforms_ = transforms.Compose([
                    transforms.ToTensor(),
                    getAugmentedTensor(self.augmentation_dict),
                    transforms.Normalize(self.means,self.stds) 
            ])
        else:
            transforms_ = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.means,self.stds) 
            ])

        data = datasets.CIFAR10(
                self.output_path, 
                train=isTrainSet_bool, 
                download=False,
                transform=transforms_)

        label_map = {0: 0, 2: 1}
        class_names = ['airplane', 'bird']
        self.cifar2 = [(img, label_map[label])
                   for img, label in data if label in [0, 2]]
             
        log.info("{} {} samples".format(
            len(self.cifar2),
            "training" if isTrainSet_bool else "validation"
        ))   

    def __len__(self):
        return len(self.cifar2)

    def __getitem__(self, ndx):
        return self.cifar2[ndx]



        