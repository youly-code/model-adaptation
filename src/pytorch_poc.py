import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Apple Silicon) available? {torch.backends.mps.is_available()}")
print(f"Is MPS built? {torch.backends.mps.is_built()}")

# First, let's get our MNIST data
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # Normalize the data (these are MNIST's mean/std)
    ]
)

# Download training data - this will be our "teaching material"
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# And some test data to see how well we're doing
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)


# Let's build our neural network!
class OurCoolNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # LAYER 1: The "Looking at the image" layer
        # Why 784? That's our 28x28 pixel image flattened into a line
        # Why 128? It's like having 128 different "pattern detectors"
        # More neurons = more patterns it can learn, but also more computation
        # 128 is a sweet spot for this problem - not too few, not too many
        self.layer1 = nn.Linear(784, 128)

        # ReLU: The "Decision Maker"
        # ReLU(x) = max(0, x)
        # It's like telling our network:
        # "If you're pretty sure about something (>0), keep that information
        #  If you're not sure (<0), let's ignore it (set to 0)"
        # This helps the network make clearer decisions!
        self.relu = nn.ReLU()

        # LAYER 2: The "Making a guess" layer
        # Takes our 128 pattern detectors and combines them into 10 final guesses
        # Why 10? One for each digit (0-9)
        # Each output tells us: "How confident am I this is a 0? A 1? A 2? etc."
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        # Step 1: Flatten the image
        # Turn 28x28 pixel image into one long line of 784 pixels
        # -1 means "figure out the batch size automatically"
        x = x.view(-1, 784)

        # Step 2: Find patterns
        # Pass through first layer to detect patterns
        # Then through ReLU to make decisive yes/no choices about those patterns
        x = self.relu(self.layer1(x))

        # Step 3: Make our final guess
        # Combine all the patterns we found into 10 confidence scores
        # Each score says how likely we think this image is that digit
        x = self.layer2(x)
        return x


# Create our network and move it to that sweet M3 GPU!
model = OurCoolNetwork().to("mps")

# Adam optimizer: Like a smart teacher that adjusts how big steps we take when learning
# lr=0.001 means "take small careful steps" - too big and we might miss the best answer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CrossEntropyLoss: How we measure wrongness
# It's like a score that says "how embarrassingly wrong were we?"
# The network tries to minimize this score
criterion = nn.CrossEntropyLoss()  # Here's our loss function!

# Let's break down what happens inside with a simple example:

# Imagine our network looking at an image of a '7'
# Our network outputs 10 numbers (confidence scores for each digit):
#              0    1    2    3    4    5    6    7    8    9
# outputs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]
#                                                ^
#                                      High confidence it's a 7!

# The ACTUAL answer (label) is 7

# CrossEntropyLoss does three main things:
# 1. Applies softmax to convert outputs to probabilities (they sum to 1)
# 2. Takes the negative log of the probability for the correct answer
# 3. The bigger the mistake, the bigger the loss!

# Example scenarios:
# GOOD PREDICTION:
# Network says: "90% sure it's a 7" (correct) → Small loss ≈ 0.1
#
# BAD PREDICTION:
# Network says: "90% sure it's a 2" (wrong!) → Big loss ≈ 2.4
#
# UNCERTAIN PREDICTION:
# Network says: "10% for each digit" → Medium loss ≈ 2.3


# Training-related stats we want to track over time
train_losses = []  # How bad our guesses are (lower is better)
train_accuracies = []  # How many we get right during training
test_accuracies = []  # How many we get right on new digits


# Function to check how well we're doing on test data
def test_network():
    model.eval()
    with torch.no_grad():
        # Get all test data at once
        images, labels = next(iter(test_loader))
        images, labels = images.to("mps"), labels.to("mps")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
    return 100 * correct / total


# Function to create pretty plots of our progress
def plot_progress(epoch, train_loss, train_acc, test_acc):
    # Print current stats
    print(f"Epoch: {epoch}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")


def show_prediction_batch(images, labels, predictions=None):
    plt.figure(figsize=(15, 5))
    for i in range(10):  # Show first 10 images
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i][0].cpu().numpy(), cmap="gray")
        true_label = labels[i].item()
        if predictions is not None:
            pred_label = predictions[i].item()
            color = "green" if pred_label == true_label else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        else:
            plt.title(f"Label: {true_label}")
    plt.tight_layout()
    plt.show()


# THE MAIN TRAINING LOOP
epochs = 10  # How many times we'll look at all training images

for epoch in range(epochs):
    model.train()  # Tell the model "time to learn!"
    running_loss = 0.0  # Track loss for this epoch
    correct = 0  # Track correct predictions
    total = 0  # Track total predictions

    # Loop through batches of images
    for batch_idx, (images, labels) in enumerate(train_loader):
        # STEP 1: PREPARATION
        # Move data to our M3 GPU
        images, labels = images.to("mps"), labels.to("mps")

        # STEP 2: RESET GRADIENTS
        # Like erasing last episode's learning before learning something new
        optimizer.zero_grad()

        # STEP 3: FORWARD PASS
        # Let the model make its best guess
        outputs = model(images)
        # Calculate how wrong we were (loss)
        loss = criterion(outputs, labels)

        # STEP 4: BACKWARD PASS
        # Figure out how to adjust our weights to do better next time
        loss.backward()

        # STEP 5: OPTIMIZATION
        # Actually adjust our weights to learn
        optimizer.step()

        # STEP 6: TRACK PROGRESS
        running_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs.data, 1)  # Get our predictions
        total += labels.size(0)  # Count total images
        correct += (predicted == labels).sum().item()  # Count correct predictions

        # Every 100 batches, show our progress!
        if batch_idx % 100 == 99:
            # Calculate current statistics
            train_loss = running_loss / 100
            train_acc = 100 * correct / total
            test_acc = test_network()  # Check accuracy on test data

            # Store for plotting
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Show progress (no plots)
            plot_progress(epoch, train_loss, train_acc, test_acc)
            show_prediction_batch(images, labels, predicted)

            # Reset counters for next batch
            running_loss = 0.0
            correct = 0
            total = 0
