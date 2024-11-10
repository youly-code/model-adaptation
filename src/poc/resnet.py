import torch
import torch.nn as nn


# Traditional Network (The old way)
class TraditionalNet(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
        self.first = nn.Linear(1, 64)
        self.last = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.first(x))
        for layer in self.layers:
            x = torch.relu(layer(x))  # Each layer transformation
        return self.last(x)


# ResNet (The superhero way! 🦸‍♂️)
class TinyResNet(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
        self.first = nn.Linear(1, 64)
        self.last = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.first(x))
        for layer in self.layers:
            identity = x  # Save the input
            x = torch.relu(layer(x))  # Transform
            x = x + identity  # The magic skip connection! 🌟
        return self.last(x)


# Let's train them!
def compare_gradients():
    traditional = TraditionalNet()
    resnet = TinyResNet()

    x = torch.randn(1, 1)  # Random input
    y = torch.randn(1, 1)  # Random target

    # One forward and backward pass
    trad_out = traditional(x)
    trad_loss = (trad_out - y).pow(2)
    trad_loss.backward()

    res_out = resnet(x)
    res_loss = (res_out - y).pow(2)
    res_loss.backward()

    # Print gradient norms at each layer
    print("Traditional Network gradients:")
    for i, layer in enumerate(traditional.layers):
        print(f"Layer {i}: {layer.weight.grad.norm().item():.6f}")

    print("\nResNet gradients:")
    for i, layer in enumerate(resnet.layers):
        print(f"Layer {i}: {layer.weight.grad.norm().item():.6f}")


if __name__ == "__main__":
    compare_gradients()
    