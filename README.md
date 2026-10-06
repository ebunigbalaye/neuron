# Artificial Neuron: Linear Regression from Scratch

A single neuron, built with NumPy only, that learns to fit a straight line to data. In other words, a linear regression model trained with gradient descent.

## What it does

The neuron predicts an output from an input using one weight and one bias:

```
y_hat = w * x + b
```

Training adjusts `w` and `b` until the line fits the data as closely as possible:

1. **Predict** `y_hat` for every training point.
2. **Measure the error** with the mean squared error (MSE):
   `loss = mean((y_hat - y)^2)`
3. **Compute the gradients** of the loss with respect to `w` and `b`:
   `dw = 2 * mean((y_hat - y) * x)` and `db = 2 * mean(y_hat - y)`
4. **Update** the parameters a small step downhill:
   `w = w - learning_rate * dw` and `b = b - learning_rate * db`
5. **Repeat** until the loss stops falling. The final `w` and `b` define the line of best fit.

Every step uses the full training set (full-batch gradient descent), and there is no activation function, so the output is a plain linear map.

## Studying it as a dynamical system

The project also treats training as a system that moves step by step. At each step it records the weight, bias, loss and gradients, then plots them for several learning rates (0.01, 0.05, 0.1). This shows how training settles for small learning rates and blows up when the learning rate is too large.

Plots are saved to `images/`:

- weight, bias and loss over training steps
- weight and loss gradients over training steps

## Project layout

```
artificial_neuron/
    math.py             # forward pass, loss, gradients, train()
    training_data.py    # the dataset
images/                 # saved plots
```

## Running it

```
pip install numpy matplotlib
```

Run the plotting script to train the neuron at each learning rate and regenerate the images.