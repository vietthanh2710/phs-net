"""
Process the data
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for imgCount in range(len(imgs)):
            imgs[imgCount] = transforms.functional.resized_crop(imgs[imgCount], i, j, h, w, self.size, self.interpolation)
        return imgs

class SkinDataset(Dataset):
    def __init__(self, images, masks,
                 transform=True, typeData = "train"):
        self.transform = transform if typeData == "train" else False  # augment data bool
        self.typeData = typeData
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15,15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask
    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask
    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            image, mask = RandomCrop((192, 256), scale=(0.8, 0.95))([image, mask])
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])
        ################### augmentation data ########################
        if self.transform:
            image, mask = self.augment(image, mask)
        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.int64)
        mask = torch.from_numpy(mask[np.newaxis])
        return image, mask, idx

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    # load the data
    images = np.load("data/images.npy")
    masks = np.load("data/masks.npy")
    # split the data
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
    # create the dataset
    train_dataset = ISICLoader(train_images, train_masks, transform=True, typeData="train")
    val_dataset = ISICLoader(val_images, val_masks, transform=False, typeData="val")

    for image, mask, idx in train_dataset:
        print(image.shape, mask.shape, idx)
        break