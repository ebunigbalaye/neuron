import numpy as np
import matplotlib.pyplot as plt
from artificial_neuron.math import train
from artificial_neuron.training_data import training_data


input_values, output_values = training_data()
x = np.asarray(input_values, dtype=float)
y = np.asarray(output_values, dtype=float)

LEARNING_RATE = 0.01  # a rate that converges, so the path is easy to follow
run = train(input_values=input_values, output_values=output_values,
            learning_rate=LEARNING_RATE)
ws = np.asarray(run[0], dtype=float)   # weight at every step
bs = np.asarray(run[1], dtype=float)   # bias at every step

# Best weight and bias (the bottom of the bowl), found directly
w_star, b_star = np.polyfit(x, y, 1)


def mse(W, B):
    """Mean squared error for every (w, b) pair in the arrays W and B."""
    return np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)


def gradient(W, B):
    """Gradient of the MSE (no 1/2) at every (w, b) pair."""
    err = W[..., None] * x + B[..., None] - y
    return 2 * np.mean(err * x, axis=-1), 2 * np.mean(err, axis=-1)



curv_w = 2 * np.mean(x ** 2)   # steepness along the w direction
curv_b = 2.0                   # steepness along the b direction
H = 2 * np.array([[np.mean(x ** 2), np.mean(x)],
                  [np.mean(x), 1.0]])
eigvals, eigvecs = np.linalg.eigh(H)

print(f"best weight  w* = {w_star:.4f}")
print(f"best bias    b* = {b_star:.4f}")
print(f"steepness along w: {curv_w:.3f}")
print(f"steepness along b: {curv_b:.3f}")
print(f"steepest / flattest direction: {eigvals.max():.3f} / {eigvals.min():.3f}")
print(f"ratio (condition number): {eigvals.max() / eigvals.min():.1f}")
print(f"predicted learning-rate limit: {2 / eigvals.max():.4f}")

span = 1.2 * max(np.abs(ws - w_star).max(), np.abs(bs - b_star).max(), 1.0)
w_axis = np.linspace(w_star - span, w_star + span, 120)
b_axis = np.linspace(b_star - span, b_star + span, 120)
W, B = np.meshgrid(w_axis, b_axis)
L = mse(W, B)
L_min = mse(np.array(w_star), np.array(b_star))
path_loss = mse(ws, bs)


fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(W, B, L, cmap="viridis", alpha=0.65, linewidth=0)
ax.plot(ws, bs, path_loss, "o-", color="#E53E3E", markersize=3,
        linewidth=1.5, label=f"training path (lr={LEARNING_RATE})")
ax.scatter([w_star], [b_star], [L_min], color="black", s=40, label="best (w*, b*)")
ax.set_xlabel("weight w")
ax.set_ylabel("bias b")
ax.set_zlabel("loss")
ax.set_title("The loss bowl", fontweight="bold")
ax.legend(loc="upper left")
fig.savefig("images/bowl_3d.png", dpi=300)
plt.show()


