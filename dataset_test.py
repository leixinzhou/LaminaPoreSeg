from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt 

prefix = "../../pre"
transform = transforms.Compose([
    ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.4
    ),
    RandomAffine(
        degrees=360, 
        translate=(0.2, 0.2),
        scale=(0.8, 1.3),
        shear=(10, 20)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomResizedCrop(
        size=(128,128),
        scale=(0.3,1.0)
    ),
    ToTensor()
])
tr_dataset = LaminaDataset(prefix=prefix, transform=transform)
tr_loader = DataLoader(dataset=tr_dataset, batch_size=1, shuffle=False)

if __name__ == '__main__':
    for step, batch in enumerate(tr_loader):
        img = batch['img'].squeeze().detach().numpy()
        gt = batch['gt'].squeeze().detach().numpy()*255
        _, ax = plt.subplots(1,2)
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(img, cmap='gray')
        ax[1].imshow(gt, cmap='jet', alpha=0.5)
        plt.show()
        break