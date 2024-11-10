import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars


class BetterBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.improve = nn.Sequential(
            # First conv - expand
            nn.Conv2d(channels, channels * 2, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            # Main conv - transform
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            # Final conv - reduce
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.improve(x))


class ModernNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # First layer - get the features started
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Modern blocks with increasing channels
        self.blocks = nn.Sequential(BetterBlock(64), BetterBlock(64), BetterBlock(64))

        # Final classification
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classify(x)


class TrainingLogger:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []

    def log(self, train_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

    def plot(self):
        plt.figure(figsize=(15, 5))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training Loss")
        plt.title("Training Loss over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label="Training Accuracy")
        plt.plot(self.val_accs, label="Validation Accuracy")
        plt.title("Accuracy over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.show()


def train_epoch(model, train_loader, val_loader, optimizer, scheduler, device, logger):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Training loop with progress bar
    train_pbar = tqdm(train_loader, desc="Training")
    for data, target in train_pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Update progress bar
        train_pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"}
        )

    train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation")
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += target.size(0)

            # Update progress bar
            val_pbar.set_postfix({"acc": f"{100. * val_correct / val_total:.2f}%"})

    val_acc = 100.0 * val_correct / val_total

    # Log metrics
    logger.log(train_loss, train_acc, val_acc)

    return train_loss, train_acc, val_acc


def main():
    # Settings
    num_epochs = 10
    batch_size = 128

    # Setup device (optimized for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data augmentation and normalization
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    valset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model, optimizer, scheduler
    model = ModernNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )

    # Create logger
    logger = TrainingLogger()

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc, val_acc = train_epoch(
            model, train_loader, val_loader, optimizer, scheduler, device, logger
        )
        print(
            f"Epoch {epoch+1}: "
            f"Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%"
        )

    # Plot training history
    logger.plot()

    # Save the model
    torch.save(model.state_dict(), "modern_net.pth")
    print("Training complete! Model saved as 'modern_net.pth'")


if __name__ == "__main__":
    main()
