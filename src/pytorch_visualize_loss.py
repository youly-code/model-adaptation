def visualize_loss():
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    # Create confidence scores for different scenarios
    correct_digit = 7
    predictions = torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1],  # Confident & Correct
            [0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Confident & Wrong
            [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1],  # Uncertain
        ]
    )

    # Calculate losses
    labels = torch.tensor([correct_digit] * 3)
    losses = F.cross_entropy(predictions, labels, reduction="none")

    # Plot
    scenarios = ["Confident & Correct", "Confident & Wrong", "Uncertain"]
    plt.figure(figsize=(10, 5))
    plt.bar(scenarios, losses.detach().numpy())
    plt.title("CrossEntropyLoss for Different Prediction Scenarios")
    plt.ylabel("Loss Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print actual values
    for scenario, loss in zip(scenarios, losses):
        print(f"{scenario}: Loss = {loss.item():.4f}")


visualize_loss()
