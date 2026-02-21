import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, dropout_p):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits, targets):
    _, preds = torch.max(logits, 1)
    correct = preds.eq(targets).sum().item()
    total = targets.size(0)
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_data_aug = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    train_data_plain = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=eval_transform)
    test_data = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=eval_transform)

    num_train = len(train_data_aug)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(args.valid_size * num_train))
    valid_idx = indices[:split]
    train_idx = indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data_aug,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    valid_loader = torch.utils.data.DataLoader(
        train_data_plain,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model = Net(args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_loss = float("inf")
    best_path = os.path.join(args.output_dir, "model_trained.pt")

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * data.size(0)
            c, t = accuracy_from_logits(output, target)
            train_correct += c
            train_total += t

        model.eval()
        valid_loss_sum = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)

                valid_loss_sum += loss.item() * data.size(0)
                c, t = accuracy_from_logits(output, target)
                valid_correct += c
                valid_total += t

        train_loss = train_loss_sum / len(train_loader.sampler)
        valid_loss = valid_loss_sum / len(valid_loader.sampler)

        train_acc = 100.0 * train_correct / max(1, train_total)
        valid_acc = 100.0 * valid_correct / max(1, valid_total)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(
            "Epoch: {} | Train Loss: {:.6f} | Val Loss: {:.6f} | Train Acc: {:.2f}% | Val Acc: {:.2f}%".format(
                epoch, train_loss, valid_loss, train_acc, valid_acc
            )
        )

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_path)
            print("Saved best model to {}".format(best_path))

    elapsed = time.time() - start_time
    print("Training time (seconds): {:.2f}".format(elapsed))

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    class_correct = [0.0 for _ in range(10)]
    class_total = [0.0 for _ in range(10)]

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, target)
            test_loss_sum += loss.item() * data.size(0)

            _, preds = torch.max(output, 1)
            correct_tensor = preds.eq(target)

            test_correct += correct_tensor.sum().item()
            test_total += target.size(0)

            preds_cpu = preds.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()
            correct_cpu = correct_tensor.detach().cpu().numpy()

            i = 0
            while i < target.size(0):
                label = int(target_cpu[i])
                class_correct[label] += float(correct_cpu[i])
                class_total[label] += 1.0
                i += 1

    test_loss = test_loss_sum / len(test_loader.dataset)
    test_acc = 100.0 * test_correct / max(1, test_total)

    print("Test Loss: {:.6f}".format(test_loss))
    for i in range(10):
        if class_total[i] > 0:
            acc_i = 100.0 * class_correct[i] / class_total[i]
            print("Test Accuracy of {:>10s}: {:>6.2f}% ({}/{})".format(classes[i], acc_i, int(class_correct[i]), int(class_total[i])))
        else:
            print("Test Accuracy of {:>10s}: N/A".format(classes[i]))

    print("Test Accuracy (Overall): {:.2f}% ({}/{})".format(test_acc, int(test_correct), int(test_total)))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_plot_path = os.path.join(args.output_dir, "loss_curves.png")
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    metrics = {
        "model": "3xConv(32,64,128)+MaxPool+ReLU+Dropout+FC(256)->10",
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "augmentation": "RandomCrop(32,pad=4) + RandomHorizontalFlip",
        "normalization": {"mean": cifar10_mean, "std": cifar10_std},
        "best_valid_loss": best_valid_loss,
        "final_test_loss": test_loss,
        "final_test_accuracy_percent": test_acc,
        "loss_curves_path": loss_plot_path,
        "device": str(device),
        "seed": args.seed,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved loss curves to {}".format(loss_plot_path))
    print("Saved metrics to {}".format(metrics_path))


if __name__ == "__main__":
    main()