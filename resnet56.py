import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.expansion = 1
        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64 * self.expansion, num_classes)
        self.relu = nn.ReLU()

        self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,
                                out_channels, stride))
            self.in_channels = out_channels * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = nn.functional.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet56(ResNet):
    def __init__(self, num_classes=10):
        super(ResNet56, self).__init__(ResNetBlock, [9, 9, 9], num_classes)


# Define data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
# This is where these normalization values come from:
# https://github.com/kuangliu/pytorch-cifar/issues/19

# Load CIFAR-10 train, valdation and test sets
dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)

classes = dataset.classes
class_count = {}
# Count number of images per class
for _, index in dataset:
    label = classes[index]
    if label not in class_count:
        class_count[label] = 0
    class_count[label] += 1
print(class_count)

val_size = 5000  # 10% of dataset
train_size = len(dataset) - val_size

trainset, validationset = torch.utils.data.random_split(
    dataset, [train_size, val_size])
print(len(trainset), len(validationset))


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
validationloader = torch.utils.data.DataLoader(
    validationset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)


def displayImageGrid(dataset):
    for images, _ in dataset:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(
            torchvision.utils.make_grid(
                images, nrow=16).permute(
                (1, 2, 0)))
        break

# displayImageGrid(trainloader)


# Hyperparameters
num_epochs = 200
initial_lr = 0.1
momentum = 0.9
weight_decay = 1e-4
device = 'cuda'
criterion = nn.CrossEntropyLoss()


model = ResNet56(num_classes=10).to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=initial_lr,
    momentum=momentum,
    weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150])

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainloader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

    # Calculate training accuracy
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        correct_train = 0
        total_train = 0
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = correct_train / total_train

    # Calculate validation loss and accuracy
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        for data in validationloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
        val_accuracy = correct_val / total_val

    # Step the learning rate scheduler based on validation accuracy
    scheduler.step()

    # Save checkpoint after each epoch during last 10 epochs
    if epoch % 10 == 9 or epoch >= num_epochs - 10:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }
        torch.save(checkpoint, f'./data/model_checkpoint_epoch{epoch+1}.pth')
    # Print epoch statistics
    print(f"Epoch {epoch+1} / {num_epochs}, "
          f"Loss: {loss.item():.4f}, "
          f"Training Accuracy: {train_accuracy*100:.2f}%, "
          f"Validation Accuracy: {val_accuracy*100:.2f}%, "
          f"Validation Loss: {val_loss:.4f}")

print("Finished Training")

# Final evaluation
model.eval()
with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for data, targets in testloader:
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
    print(f"Accuracy: {float(num_correct)/float(num_samples)*100:.2f}")
