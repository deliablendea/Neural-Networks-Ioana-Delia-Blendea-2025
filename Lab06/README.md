# Lab06: Overfitting, Activations, Optimizers, and Regularization

This lab covers:

- Overfitting: visualize train vs. validation behavior on a simple dataset
- Activation functions: Sigmoid, ReLU, GELU (with derivatives and intuition)
- Optimizers: SGD, Momentum, RMSprop, Adam — implemented from scratch and visualized on a clear 1D curve; plus a 2D Rosenbrock demo that shows why Momentum/Adam can outperform plain SGD.
- Regularization: L1/L2 penalties, Dropout, BatchNorm, LayerNorm, Data Augmentation

What you’ll do:

- Implement and compare optimizer update rules without using `torch.optim`
- Plot optimizer steps on the 1D function curve, and on a 2D Rosenbrock contour to see zig‑zag vs smooth descent
- Train simple MLPs with different activations and regularizations
- Track and visualize parameter norms over training for L1/L2
- Implement Dropout, BatchNorm, and LayerNorm forward passes using basic tensor ops
- Build simple image augmentations with tensor ops (no torchvision transforms)

Open `Lab06/Lab06.ipynb` and run cells top to bottom. The notebook uses only basic PyTorch operations and Matplotlib for clear, step-by-step visualizations.

Tip: If you’re short on time in class, run the “Optimizers on a 1D curve” (then the Rosenbrock demo) and “Activations” sections first — they’re fast, visual, and self-contained.

Requirements:

- Python 3.8+
- PyTorch 1.12+ (CPU is fine), NumPy, Matplotlib, Pillow

Notes:

- The augmentation demo loads `Lab04/relu.png` if present; it falls back to a synthetic image otherwise.
- We implement optimizers, Dropout, BatchNorm, and LayerNorm using basic tensor ops (no `torch.optim` or built-in layers). Autograd is used for gradients.
