import torch
import torch.nn as nn

# Training data
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])

y = torch.tensor([[3.0],
                  [5.0],
                  [7.0],
                  [9.0]])

# Neural network
model = nn.Sequential(
    nn.Linear(1, 1)
)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# Training loop
for epoch in range(1000):

    # Forward pass
    prediction = model(x)

    # Calculate error
    loss = loss_fn(prediction, y)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

print(model(torch.tensor([[5.0]])))
