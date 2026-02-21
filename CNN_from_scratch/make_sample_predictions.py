import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    os.makedirs("outputs", exist_ok=True)

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_data = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device("cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load("outputs/model_trained.pt", map_location=device))
    model.eval()

    images, labels = next(iter(test_loader))

    with torch.no_grad():
        output = model(images.to(device))
        _, preds = torch.max(output, 1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    n = images.size(0)
    if n > 20:
        n = 20

    fig = plt.figure(figsize=(25, 4))
    i = 0
    while i < n:
        ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
        img = images[i] * std_t + mean_t
        img = torch.clamp(img, 0.0, 1.0)
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        p = int(preds[i].item())
        t = int(labels[i].item())
        color = "green" if p == t else "red"
        ax.set_title("{} ({})".format(classes[p], classes[t]), color=color)
        i += 1

    out_path = "outputs/sample_predictions.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(out_path))

if __name__ == "__main__":
    main()