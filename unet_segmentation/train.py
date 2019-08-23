import torch
import torch.nn as nn
import torch.optim as optim

# Loading & Normalize Dataset
from unet_segmentation.dataset_unet import LinemodDataset
from unet_segmentation.unet import UNet

trainset = LinemodDataset(mode="train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

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
dir_checkpoint = trainset.root_folder

# net.load_state_dict(torch.load(dir_checkpoint+"3CP15.pth"))

# TRAIN NN
for epoch in range(60):  # loop over the dataset multiple times

    running_loss = 0.0
    all_loss = 0.0

    for i, data in enumerate(trainloader, 100):
        # get the inputs
        inputs, labels = data  # inputs [batchsize, h, w, c]; labels [batchsize, h, w]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        all_loss += loss.item()
        if i % 25 == 0:  # print every 25 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 25))
            running_loss = 0.0

    if True:
        torch.save(
            net.state_dict(),
            dir_checkpoint + 'HD_CP{}.pth'.format(epoch + 1)
        )
        print('Checkpoint {} saved !'.format(epoch + 1))
        print("LOSS FOR THIS EPOCH: ", all_loss)

print('Finished Training')

