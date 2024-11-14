import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List


class NetworkType:
    RESNET = "resnet"
    TRADITIONAL = "traditional"


class EnhancedNet(nn.Module):
    def __init__(self, num_layers=5, net_type=NetworkType.RESNET):
        super().__init__()
        self.net_type = net_type
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
        self.first = nn.Linear(1, 64)
        self.last = nn.Linear(64, 1)
        self.activations: Dict[str, torch.Tensor] = {}

    def forward(self, x):
        x = torch.relu(self.first(x))
        self.activations["input"] = x.detach().mean()

        for i, layer in enumerate(self.layers):
            if self.net_type == NetworkType.RESNET:
                identity = x
                x = layer(x)
                x = x + identity  # Skip connection
                x = F.relu(x)
            else:
                x = F.relu(layer(x))

            self.activations[f"layer_{i}"] = x.detach().mean()

        return self.last(x)


def train_and_compare(batch_size=512, epochs=100):  # Made this a standalone function
    # Create networks
    resnet = EnhancedNet(net_type=NetworkType.RESNET)
    traditional = EnhancedNet(net_type=NetworkType.TRADITIONAL)

    # Optimizers
    optim_res = torch.optim.Adam(resnet.parameters(), lr=0.001)
    optim_trad = torch.optim.Adam(traditional.parameters(), lr=0.001)

    # Training history
    history = {
        "resnet_loss": [],
        "traditional_loss": [],
        "resnet_activations": {f"layer_{i}": [] for i in range(5)},
        "traditional_activations": {f"layer_{i}": [] for i in range(5)},
    }

    # Training loop
    for epoch in range(epochs):
        # Generate data
        x = torch.randn(batch_size, 1)  # Fixed: Added batch_size
        y = x.pow(2)  # Quadratic function to learn

        # Train ResNet
        optim_res.zero_grad()
        out_res = resnet(x)
        loss_res = F.mse_loss(out_res, y)
        loss_res.backward()
        optim_res.step()

        # Train Traditional
        optim_trad.zero_grad()
        out_trad = traditional(x)
        loss_trad = F.mse_loss(out_trad, y)
        loss_trad.backward()
        optim_trad.step()

        # Store history
        history["resnet_loss"].append(loss_res.item())
        history["traditional_loss"].append(loss_trad.item())

        # Store activations
        for i in range(5):
            history["resnet_activations"][f"layer_{i}"].append(
                resnet.activations[f"layer_{i}"].item()
            )
            history["traditional_activations"][f"layer_{i}"].append(
                traditional.activations[f"layer_{i}"].item()
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"ResNet Loss: {loss_res.item():.6f}")
            print(f"Traditional Loss: {loss_trad.item():.6f}\n")

    return history


def setup_plot_axes(ax, title: str, y_label: str):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)


def visualize_results(history):
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Loss Comparison
    ax1.plot(history["resnet_loss"], label="ResNet", color="blue", alpha=0.7)
    ax1.plot(history["traditional_loss"], label="Traditional", color="red", alpha=0.7)
    setup_plot_axes(ax1, "ResNet vs Traditional Network Training Loss", "Loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Layer Activations
    for i in range(5):
        ax2.plot(
            history["resnet_activations"][f"layer_{i}"],
            label=f"ResNet Layer {i}",
            linestyle="-",
        )
        ax2.plot(
            history["traditional_activations"][f"layer_{i}"],
            label=f"Trad Layer {i}",
            linestyle="--",
        )

    setup_plot_axes(ax2, "Layer Activations Over Time", "Mean Activation")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Train both networks and get history
    print("Starting training...")
    history = train_and_compare(batch_size=512, epochs=100)

    # Visualize the results
    print("\nCreating visualizations...")
    visualize_results(history)


if __name__ == "__main__":
    main()
