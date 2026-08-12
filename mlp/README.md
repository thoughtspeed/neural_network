# Multilayer Perceptron (MLP)

Minimal feedforward MLP implemented with PyTorch. The model is defined in
`multilayer_perceptron.py` as `NeuralNetwork(num_inputs, num_outputs)`.

Overview
- Simple MLP with two hidden layers (30 and 20 neurons) and ReLU activations.
- The network returns raw logits from `forward()`; choose loss accordingly:
  - Classification: set `num_outputs` = number of classes and use
    `nn.CrossEntropyLoss()` (expects integer class labels and raw logits).
  - Regression: set `num_outputs` = target dimensionality and use
    `nn.MSELoss()`.

Quickstart

Requirements
- Python 3.8+
- PyTorch (install via https://pytorch.org/get-started/locally or `pip install torch`)

Run example
```bash
cd neural_network/multilayer_perceptron
python multilayer_perceptron.py
```
This prints the model structure and the first layer weight shape and basic stats.

Smoke test (python REPL or script)
```python
import torch
from multilayer_perceptron import NeuralNetwork

model = NeuralNetwork(50, 3)
batch = torch.randn(4, 50)
out = model(batch)
assert out.shape == (4, 3)
```

One-line training example (classification)
```python
# single optimization step: forward, loss, backward, step
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2); loss_fn = torch.nn.CrossEntropyLoss(); x = torch.randn(8, 50); y = torch.randint(0, 3, (8,)); optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
```

Weights and shapes
- `nn.Linear(in_features=50, out_features=30)` stores `weight` with shape
  `(30, 50)` because PyTorch uses `(out_features, in_features)`. When a batch
  `x` has shape `(batch_size, 50)`, the layer computes `x @ weight.T + bias`
  producing output `(batch_size, 30)`.

Save / load example
```python
torch.save(model.state_dict(), "mlp.pth")
model = NeuralNetwork(50, 3)
model.load_state_dict(torch.load("mlp.pth"))
```
