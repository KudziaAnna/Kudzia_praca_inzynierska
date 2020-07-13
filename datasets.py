import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from Prepering_data import get_dataset

class DHDataset(Dataset):
    """Brain dMRI dataset for harmonization"""

    def __init__(self, b_x, b_y, transform = None):
        'Initialization'
        self.input_images, self.target_masks = get_dataset(b_x, b_y)
        self.transform = transform


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_images)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = self.input_images[index]
        mask = self.target_masks[index]

        return [image, mask]


def load_data(b_x, b_y, batch_size):
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        #transforms.Resize(128),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()  
    ])

    data = DHDataset(b_x, b_y, transform= trans)

    indices = np.random.permutation(len(data))

    train_set = data[indices[:int(len(data)*0.8) ] ]
    val_set = data[indices[int(len(data)*0.8) : ] ]

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    return dataloaders