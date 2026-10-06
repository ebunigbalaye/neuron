from artificial_neuron.math import train
from artificial_neuron.training_data import training_data
import matplotlib.pyplot as plt

input_values, output_values = training_data()
small_learning_rate = train(input_values=input_values,output_values=output_values,learning_rate=0.01)
medium_learning_rate = train(input_values=input_values,output_values=output_values,learning_rate=0.05)
large_learning_rate = train(input_values=input_values,output_values=output_values,learning_rate=0.1)



steps = range(0, 100)
weights = [small_learning_rate[0], medium_learning_rate[0], large_learning_rate[0]]
learning_rates = [0.01, 0.05, 0.1]
colours = ["#E53E3E", "#38A169", "#4299E1"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, layout='tight')

for ax, lr, ws, colour in zip(axes, learning_rates, weights, colours):
    ax.plot(steps, ws, 'o-', markersize=3, color=colour, linewidth=1.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Learning rate {lr}", fontsize=13, fontweight='bold', color="#2D3748")
    ax.set_xlabel("Step", fontsize=12, color="#4A5568", labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E0')
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E2E8F0")
    ax.tick_params(colors="#718096", labelsize=10)

axes[0].set_ylabel("Weight", fontsize=12, color="#4A5568", labelpad=8)
fig.suptitle("Visualizing Weight Dynamics", fontsize=16, fontweight='bold', color="#2D3748")
plt.savefig("images/weight_plot.png", dpi=300)


steps = range(0, 100)
weights = [small_learning_rate[1], medium_learning_rate[1], large_learning_rate[1]]
learning_rates = [0.01, 0.05, 0.1]
colours = ["#E53E3E", "#38A169", "#4299E1"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, layout='tight')

for ax, lr, ws, colour in zip(axes, learning_rates, weights, colours):
    ax.plot(steps, ws, 'o-', markersize=3, color=colour, linewidth=1.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Learning rate {lr}", fontsize=13, fontweight='bold', color="#2D3748")
    ax.set_xlabel("Step", fontsize=12, color="#4A5568", labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E0')
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E2E8F0")
    ax.tick_params(colors="#718096", labelsize=10)

axes[0].set_ylabel("Bias", fontsize=12, color="#4A5568", labelpad=8)
fig.suptitle("Visualizing Bias Dynamics", fontsize=16, fontweight='bold', color="#2D3748")

plt.savefig("images/bias_plot.png", dpi=300)
plt.close(fig)



steps = range(0, 100)
weights = [small_learning_rate[2], medium_learning_rate[2], large_learning_rate[2]]
learning_rates = [0.01, 0.05, 0.1]
colours = ["#E53E3E", "#38A169", "#4299E1"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, layout='tight')

for ax, lr, ws, colour in zip(axes, learning_rates, weights, colours):
    ax.plot(steps, ws, 'o-', markersize=3, color=colour, linewidth=1.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Learning rate {lr}", fontsize=13, fontweight='bold', color="#2D3748")
    ax.set_xlabel("Step", fontsize=12, color="#4A5568", labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E0')
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E2E8F0")
    ax.tick_params(colors="#718096", labelsize=10)

axes[0].set_ylabel("Loss", fontsize=12, color="#4A5568", labelpad=8)
fig.suptitle("Visualizing Loss Dynamics", fontsize=16, fontweight='bold', color="#2D3748")
plt.savefig("images/loss_plot.png", dpi=300)



steps = range(0, 100)
weights = [small_learning_rate[3], medium_learning_rate[3], large_learning_rate[3]]
learning_rates = [0.01, 0.05, 0.1]
colours = ["#E53E3E", "#38A169", "#4299E1"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, layout='tight')

for ax, lr, ws, colour in zip(axes, learning_rates, weights, colours):
    ax.plot(steps, ws, 'o-', markersize=3, color=colour, linewidth=1.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Learning rate {lr}", fontsize=13, fontweight='bold', color="#2D3748")
    ax.set_xlabel("Step", fontsize=12, color="#4A5568", labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E0')
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E2E8F0")
    ax.tick_params(colors="#718096", labelsize=10)

axes[0].set_ylabel("Weight Gradient", fontsize=12, color="#4A5568", labelpad=8)
fig.suptitle("Visualizing Weight Gradient Dynamics", fontsize=16, fontweight='bold', color="#2D3748")
plt.savefig("images/weight_gradient_plot.png", dpi=300)



steps = range(0, 100)
weights = [small_learning_rate[4], medium_learning_rate[4], large_learning_rate[4]]
learning_rates = [0.01, 0.05, 0.1]
colours = ["#E53E3E", "#38A169", "#4299E1"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, layout='tight')
for ax, lr, ws, colour in zip(axes, learning_rates, weights, colours):
    ax.plot(steps, ws, 'o-', markersize=3, color=colour, linewidth=1.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Learning rate {lr}", fontsize=13, fontweight='bold', color="#2D3748")
    ax.set_xlabel("Step", fontsize=12, color="#4A5568", labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E0')
    ax.spines['bottom'].set_color('#CBD5E0')
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E2E8F0")
    ax.tick_params(colors="#718096", labelsize=10)

axes[0].set_ylabel("Bias Gradient", fontsize=12, color="#4A5568", labelpad=8)
fig.suptitle("Visualizing Bias Gradient Dynamics", fontsize=16, fontweight='bold', color="#2D3748")
plt.savefig("images/bias_gradient_plot.png", dpi=300)
plt.close(fig)
