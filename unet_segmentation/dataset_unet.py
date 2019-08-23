import os

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps

root_folder = "/home/lorenzo/PycharmProjects/DenseFusionNew/datasets/linemod/Linemod_preprocessed/data/06"


class LinemodDataset(data.Dataset):
    def __init__(self, mode, root=root_folder):
        self.root_folder = root
        self.root_rgb = root + "/rgb/"
        self.root_label = root + "/mask/"

        self.list_rgb = os.listdir(self.root_rgb)
        self.list_label = os.listdir(self.root_label)

        self.list_rgb = sorted(list(set(self.list_rgb) & set(self.list_label)))
        self.list_label = self.list_rgb

        self.mode = mode

        # 80 - 20 split
        if mode == "train":
            self.list_rgb = self.list_rgb[0: 1000]
            self.list_label = self.list_label[0: 1000]
            self.length = len(self.list_rgb) * 6

        elif mode == "test":
            self.list_rgb = self.list_rgb[1000: len(self.list_rgb)]
            self.list_label = self.list_label[1000: len(self.list_label)]
            self.length = len(self.list_rgb)

    def __getitem__(self, index):
        t = 0
        if (self.mode == "train") and (index >= len(self.list_rgb)):
            t = int(index / len(self.list_rgb))
            index = index % len(self.list_rgb)
        img = Image.open(self.root_rgb + self.list_rgb[index])
        label = Image.open(self.root_label + self.list_label[index]).convert('L')
        # converting to grayscale because of output of network

        # Data augmentation based on t

        if t == 1:
            # h axis
            img = ImageOps.mirror(img)
            label = ImageOps.mirror(label)

        if t == 2:
            # v axis
            img = img.crop((160, 120, 480, 360))
            label = label.crop((160, 120, 480, 360))
            img = ImageOps.mirror(img)
            label = ImageOps.mirror(label)
            img = img.resize((640, 480))
            label = label.resize((640, 480))

        if t == 3:
            # cropping
            img = img.crop((160, 120, 480, 360))
            label = label.crop((160, 120, 480, 360))
            img = img.resize((640, 480))
            label = label.resize((640, 480))

        if t == 4:
            # cropping
            img = img.crop((80, 60, 560, 420))
            label = label.crop((80, 60, 560, 420))
            img = img.resize((640, 480))
            label = label.resize((640, 480))

        if t == 5:
            # cropping
            img = img.crop((80, 60, 560, 420))
            label = label.crop((80, 60, 560, 420))
            img = ImageOps.mirror(img)
            label = ImageOps.mirror(label)
            img = img.resize((640, 480))
            label = label.resize((640, 480))

        img = (np.array(img).astype(np.float32) - 127.5) / 255
        label = np.array(label).astype(np.float32) / 255

        return img.reshape((3, img.shape[0], img.shape[1])), label.reshape((1, label.shape[0], label.shape[1]))

    def __len__(self):
        return self.length

