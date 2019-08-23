import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw

from unet_segmentation.dataset_unet import LinemodDataset
from unet_segmentation.unet import UNet


def drawnRectangle(outputs, inp_seg):
    result = np.where(outputs == 1)
    y0 = np.min(result[0])
    y1 = np.max(result[0])
    x1 = np.max(result[1])
    x0 = np.min(result[1])
    draw = ImageDraw.Draw(inp_seg)
    for i in range(0, 4):
        draw.rectangle([(x0 + i, y0 + i), (x1 + i, y1 + i)], outline="blue")
    del (draw)
    return


testset = LinemodDataset(mode="train")
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
trainset = LinemodDataset(mode="train")

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define NN
net = UNet(n_channels=3, n_classes=1)
net.to(device)

# Define Loss & optimizer
optimizer = optim.SGD(net.parameters(),
                      lr=0.1,
                      momentum=0.9,
                      weight_decay=0.0005)

criterion = nn.BCELoss()
net.load_state_dict(torch.load("/home/lorenzo/PycharmProjects/DenseFusionMasterAI/unet_segmentation/trained_models/4CP10.pth"))

for i, data in enumerate(testloader, 0):
    if i < 5:
        inputs, labels = data
        print(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        outputs = outputs.cpu().detach().numpy()[0, 0, :, :] > 0.49
        labels = labels.cpu().detach().numpy()[0, 0, :, :]
        inputs = (inputs.cpu().detach().numpy()[0, :, :, :]).reshape((240, 320, 3))

        out = Image.fromarray(np.uint8(outputs * 255))
        lab = Image.fromarray(np.uint8(labels * 255))
        inp = Image.fromarray(np.uint8(inputs * 255 + 127.5))

        inp_seg = Image.fromarray(np.uint8(inputs * 255 + 127.5))
        drawnRectangle(outputs, inp_seg)

        plt.figure(1)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(223)
        plt.imshow(lab)
        plt.subplot(224)
        plt.imshow(out)
        plt.subplot(222)
        plt.imshow(inp_seg)
        plt.show()

    else:
        assert False

