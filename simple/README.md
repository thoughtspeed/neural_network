
# Simple Neural Network (Linear Regression) Example

This small example demonstrates a minimal PyTorch linear model trained to learn the mapping y = 2x + 1 from four sample points.

Files
- `neural_network.py`: training script that fits a single linear layer to the dataset and prints a prediction for input 5.0.

Requirements
- Python 3.8+ and PyTorch.

Install
- Follow the official PyTorch installation instructions for your platform: https://pytorch.org/get-started/locally
- Or install a CPU-only build quickly with pip (may not be the latest/optimal wheel for your system):

```bash
pip install torch
```

Run

From the repository root run:

```bash
cd neural_network/simple
python neural_network.py
```

Expected output
- The model is trained to approximate y = 2x + 1, so the printed prediction for input 5.0 should be approximately `[[11.0]]` (a Tensor near 11).

Notes
- Hyperparameters in `neural_network.py`: 1000 epochs, learning rate `0.01` using SGD and `MSELoss`.
- Try adjusting `lr` or `epoch` count in `neural_network.py` to observe training changes.

