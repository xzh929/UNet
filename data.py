import torch
from torch.utils.data import Dataset
import os
import torchvision
from PIL import Image
from torchvision.utils import save_image

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


class MaskDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        bg1 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
        bg2 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
        png_name = self.name[item]
        jpg_name = png_name[:-3] + 'jpg'
        jpg_path = os.path.join(self.path, 'JPEGImages')
        png_path = os.path.join(self.path, 'SegmentationClass')
        jpg = Image.open(os.path.join(jpg_path, jpg_name))
        png = Image.open(os.path.join(png_path, png_name))
        jpg_size = torch.Tensor(jpg.size)
        max_index = torch.argmax(jpg_size)
        factor = 256 / jpg_size[max_index]
        jpg_use = jpg.resize((jpg_size * factor).long())
        png_use = png.resize((jpg_size * factor).long())
        bg1.paste(jpg_use, (0, 0))
        bg2.paste(png_use, (0, 0))

        return transforms(bg1), transforms(bg2)


if __name__ == '__main__':
    dataset = MaskDataset(r'D:\UNET\VOC2012')
    i = 1
    for jpg, png in dataset:
        save_image(jpg, 'data/{}.jpg'.format(i))
        save_image(png, 'data/{}.png'.format(i))
        i += 1
