import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

cuda = torch.cuda.is_available()

class SiameseNetworkDataset(Dataset):
    def __init__(self,txtfilepath,transform=None):
        self.listofimages = []
        with open (txtfilepath) as f:
            for i in f.readlines():
                self.listofimages.append(i[:-1])
        self.transform = transform

    def __getitem__(self,index):
        pair_tuple = eval(self.listofimages[index])

        img0 = Image.open("./data/celeba/img_align_celeba/" + pair_tuple[0])
        img1 = Image.open("./data/celeba/img_align_celeba/" + pair_tuple[1])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array(list(pair_tuple[2]), dtype=np.float32))

    def __len__(self):
        return len(self.listofimages)
    
data = SiameseNetworkDataset('data/siamese_celeba.txt', transform = transforms.Compose([transforms.Resize((105, 105)), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype = torch.float32)]))

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(data, batch_size=32, shuffle=True, **kwargs)