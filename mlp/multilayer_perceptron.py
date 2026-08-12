import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """Simple feedforward neural network.

    Args:
        num_inputs (int): Number of input features.
        num_outputs (int): Number of outputs. For classification set this to the
            number of classes — the model returns raw logits which are suitable
            for use with `nn.CrossEntropyLoss`. For regression set this to the
            target dimensionality and use `nn.MSELoss`.
    """
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            # Deep neural network with 2 hidden layers
            # Hidden layers with 30 and 20 neurons
            nn.Linear(num_inputs, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

if __name__ == "__main__":
    torch.manual_seed(123)

    # Input layer with 50 neurons and output layer with 3 neurons
    model = NeuralNetwork(50, 3)
    print(model)

    # PyTorch stores the weight matrix as: [out_features, in_features]
    print(model.layers[0].weight.shape)

    print(model.layers[0].weight)