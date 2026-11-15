"""Train surrogate CNNs on CIFAR-10 for use in a transfer-based black-box attack.

Usage:
    python train.py --arch resnet9 --epochs 18 --out resnet9.pt
    python train.py --arch vgg --epochs 18 --out vgg.pt
"""
import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from models import ResNet9, VGGSmall

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_loaders(data_dir, batch_size):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_set = torchvision.datasets.ImageFolder(root=f"{data_dir}/cifar10/train", transform=train_tf)
    test_set = torchvision.datasets.ImageFolder(root=f"{data_dir}/cifar10/test", transform=test_tf)

    # ImageFolder sorts classes alphabetically, which happens to match CLASS_NAMES already;
    # assert this so label indices line up with the notebook's class_names ordering.
    assert train_set.classes == CLASS_NAMES, train_set.classes

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet9", "vgg"], required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    model = (ResNet9() if args.arch == "resnet9" else VGGSmall()).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                 steps_per_epoch=len(train_loader))
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"[{args.arch}] epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  test_acc={acc:.4f}  "
              f"time={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
