import torch

from unet_segmentation.dataset_unet import LinemodDataset
from unet_segmentation.unet import UNet

testset = LinemodDataset(mode="test")
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
dir_checkpoint = testset.root_folder

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define NN
net = UNet(n_channels=3, n_classes=1)
net.to(device)

net.load_state_dict(torch.load(dir_checkpoint + "4CP10.pth"))

tot_accuracy = 0

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)

    outputs = outputs.cpu().detach().numpy()[0, 0, :, :] > 0.50
    labels = labels.cpu().detach().numpy()[0, 0, :, :]

    diff = -((outputs == labels) - 1)
    # n_wrongly_class_px = torch.sum(diff) # gives back a scalar
    n_wrongly_class_px = diff.sum()  # fp + fn
    n_correctly_class_px = (diff.shape[0] * diff.shape[1]) - n_wrongly_class_px

    accuracy = n_correctly_class_px / (n_correctly_class_px + n_wrongly_class_px)

    tot_accuracy = tot_accuracy + accuracy

print("Accuracy over test set: ", tot_accuracy / i)
