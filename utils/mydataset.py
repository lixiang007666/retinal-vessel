import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MyDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        import numpy as np
        image = torch.from_numpy(self.data[0][index].transpose((2,0,1)))
        image1 = torch.from_numpy(self.data[1][index])
        image2 = torch.from_numpy(self.data[2][index])
        image3 = torch.from_numpy(self.data[3][index])
        # if self.transform is not None:
        #     image = self.transform(image)
        #     image1 = self.transform(image1)#.type(torch.long)
        #     image2 = self.transform(image2)#.type(torch.long)
        #     image3 = self.transform(image3).type(torch.long)

        return image, image1, image2, image3