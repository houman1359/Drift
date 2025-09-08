import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr

# ----------------------------
# Helper Functions
# ----------------------------

def generate_correlated_binary(n_neurons, T, e, rho):
    """
    Generate correlated binary excitability factors for neurons.

    Parameters:
    - n_neurons: Number of neurons (e.g., 2)
    - T: Number of time steps
    - e: Excitability rate (probability of zeta=1)
    - rho: Desired correlation between excitability factors

    Returns:
    - zeta: Binary array of shape (n_neurons, T)
    """
    # Mean and covariance for latent variables
    mean = np.zeros(n_neurons)
    cov = np.full((n_neurons, n_neurons), rho)
    np.fill_diagonal(cov, 1)

    # Generate latent variables
    latent = multivariate_normal.rvs(mean=mean, cov=cov, size=T)
    if n_neurons == 1:
        latent = latent.reshape(-1, 1)

    # Determine threshold for excitability rate 'e'
    # Assuming latent variables are standard normal
    threshold = np.percentile(latent, 100 * (1 - e), axis=0)

    # Generate zeta based on threshold
    zeta = (latent > threshold).astype(int).T  # Shape: (n_neurons, T)

    return zeta

def compute_cost(y, x):
    """
    Compute the similarity matching cost function.

    Parameters:
    - y: Output array of shape (n_neurons, T)
    - x: Input array of shape (T,)

    Returns:
    - cost: Scalar cost value
    """
    S_x = np.outer(x, x)  # Shape: (T, T)
    S_y = y.T @ y          # Shape: (T, T)
    cost = np.sum((S_x - S_y)**2) / 2
    return cost

# ----------------------------
# Simulation Parameters
# ----------------------------

T = 1000  # Number of time steps
eta = 0.01  # Learning rate
e = 0.5  # Excitability rate
sigma_w = 0.05  # Synaptic noise std for feedforward weights
sigma_m = 0.05  # Synaptic noise std for recurrent weights
beta_window = 10  # Window size for GLM fitting
rho_zeta_values = [0.0, 0.2, 0.4, 0.6, 0.8]  # Different excitability correlations

# Initialize input signal (fixed for drift computation)
np.random.seed(42)  # For reproducibility
x_t = np.random.normal(0, 1, T)
fixed_x = x_t.copy()  # Keeping x_t fixed as per user instruction

# Initialize lists to store simulation results
results = {
    'rho_zeta': [],
    'mean_drift_syn': [],
    'mean_drift_exc': [],
    'mean_drift_diff_syn': [],
    'mean_drift_diff_exc': [],
    'mean_beta_drift_syn': [],
    'mean_beta_drift_exc': [],
    'mean_noise_corr_syn': [],
    'mean_noise_corr_exc': [],
    'mean_change_noise_corr_syn': [],
    'mean_change_noise_corr_exc': [],
    'final_cost_syn': [],
    'final_cost_exc': []
}

# ----------------------------
# Simulation Loop
# ----------------------------

for rho_zeta in rho_zeta_values:
    print(f"Running simulation for rho_zeta = {rho_zeta}")

    # ----- Synaptic Noise Model -----
    # Initialize weights and outputs
    w_syn = np.random.normal(0, 1, 2)  # Feedforward weights
    M_syn = np.eye(2) + np.random.normal(0, 0.1, (2, 2))  # Recurrent weights, near identity
    y_syn = np.zeros((2, T))
    beta_syn = np.zeros((2, T))
    epsilon_syn = np.zeros((2, T))

    # Compute initial beta coefficients
    try:
        beta_syn[:, 0] = np.linalg.inv(M_syn) @ w_syn
    except np.linalg.LinAlgError:
        # In case M_syn is singular, add a small regularization term
        beta_syn[:, 0] = np.linalg.inv(M_syn + 1e-6 * np.eye(2)) @ w_syn

    # ----- Excitability Modulation Model -----
    # Initialize weights and outputs
    w_exc = w_syn.copy()  # Fixed synaptic weights
    M_exc = M_syn.copy()  # Fixed recurrent weights
    y_exc = np.zeros((2, T))
    y_exc_mod = np.zeros((2, T))  # Initialize y_exc_mod as 2D array
    beta_exc = np.zeros((2, T))
    epsilon_exc = np.zeros((2, T))

    # Compute initial beta coefficients
    beta_exc[:, 0] = beta_syn[:, 0]

    # Generate correlated excitability factors
    zeta = generate_correlated_binary(2, T, e, rho_zeta)  # Shape: (2, T)

    # Initialize storage for drift and correlations
    drift_syn = []
    drift_diff_syn = []
    drift_exc = []
    drift_diff_exc = []
    beta_drift_syn = []
    beta_drift_exc = []
    noise_corr_syn = []
    noise_corr_exc = []
    noise_corr_change_syn = []
    noise_corr_change_exc = []
    cost_syn = []
    cost_exc = []

    for t in range(T):
        # ----- Synaptic Noise Model -----
        # Calculate output using steady-state solution y = M^{-1} w x_t
        try:
            y_current_syn = np.linalg.inv(M_syn) @ w_syn * fixed_x[t]
        except np.linalg.LinAlgError:
            # In case M_syn is singular, add a small regularization term
            y_current_syn = np.linalg.inv(M_syn + 1e-6 * np.eye(2)) @ w_syn * fixed_x[t]
        y_syn[:, t] = y_current_syn

        # GLM Fit: Linear regression using a sliding window of 10 steps
        if t >= beta_window:
            X_syn = fixed_x[t-beta_window:t].reshape(-1, 1)  # Inputs, shape (beta_window, 1)
            Y_syn_window = y_syn[:, t-beta_window:t]  # Outputs, shape (2, beta_window)
            for i in range(2):
                # Perform least squares regression for each neuron
                beta_syn[i, t], _, _, _ = np.linalg.lstsq(X_syn, Y_syn_window[i, :], rcond=None)
        else:
            if t > 0:
                beta_syn[:, t] = beta_syn[:, t-1]
            else:
                beta_syn[:, t] = np.zeros(2)

        # Residuals
        epsilon_syn[:, t] = y_syn[:, t] - beta_syn[:, t] * fixed_x[t]

        # Drift in beta coefficients over the window
        if t >= beta_window:
            delta_beta_syn = beta_syn[:, t] - beta_syn[:, t-beta_window]
            beta_drift_syn.append(np.mean(np.abs(delta_beta_syn)))
        else:
            beta_drift_syn.append(0)

        # Noise correlation over the window
        if t >= beta_window:
            # Compute residuals over the window
            epsilon_window_syn = epsilon_syn[:, t-beta_window+1:t+1]  # Shape: (2, beta_window)
            # Compute pairwise correlations between residuals at consecutive time steps
            rho_epsilon_syn_list = []
            for w in range(beta_window - 1):
                if np.std(epsilon_window_syn[:, w]) > 1e-6 and np.std(epsilon_window_syn[:, w+1]) > 1e-6:
                    rho = np.corrcoef(epsilon_window_syn[:, w], epsilon_window_syn[:, w+1])[0, 1]
                else:
                    rho = 0
                rho_epsilon_syn_list.append(rho)
            # Average noise correlation over the window
            avg_rho_syn = np.mean(rho_epsilon_syn_list)
            noise_corr_syn.append(avg_rho_syn)
            # Change in noise correlation over the window
            if t >= 2 * beta_window:
                delta_rho_syn = abs(avg_rho_syn - noise_corr_syn[-2])
                noise_corr_change_syn.append(delta_rho_syn)
            else:
                noise_corr_change_syn.append(0)
        else:
            noise_corr_syn.append(0)
            noise_corr_change_syn.append(0)

        # Drift in outputs (Synaptic Noise Model) over the window
        if t >= beta_window:
            drift = np.var(y_syn[:, t] - y_syn[:, t-beta_window])
            drift_syn.append(drift)
            # Drift difference between neurons
            drift_diff = np.var((y_syn[0, t] - y_syn[0, t-beta_window]) - (y_syn[1, t] - y_syn[1, t-beta_window]))
            drift_diff_syn.append(drift_diff)
        else:
            drift_syn.append(0)
            drift_diff_syn.append(0)

        # Compute cost function
        cost_current_syn = compute_cost(y_syn[:, :t+1], fixed_x[:t+1])
        cost_syn.append(cost_current_syn)

        # Synaptic updates with noise
        # Synaptic Noise Model: Independent noise
        xi_w_syn = np.random.normal(0, sigma_w, 2)
        xi_M_syn = np.random.normal(0, sigma_m, (2, 2))

        delta_w_syn = eta * (y_current_syn * fixed_x[t] - w_syn) + xi_w_syn
        delta_M_syn = eta * (np.outer(y_current_syn, y_current_syn) - M_syn) + xi_M_syn

        w_syn += delta_w_syn
        M_syn += delta_M_syn

        # ----- Excitability Modulation Model -----
        # Calculate output using steady-state solution y = M^{-1} w x_t
        try:
            y_current_exc = np.linalg.inv(M_exc) @ w_exc * fixed_x[t]
        except np.linalg.LinAlgError:
            # In case M_exc is singular, add a small regularization term
            y_current_exc = np.linalg.inv(M_exc + 1e-6 * np.eye(2)) @ w_exc * fixed_x[t]
        y_exc[:, t] = y_current_exc

        # Apply excitability modulation
        y_exc_mod[:, t] = y_exc[:, t] * zeta[:, t]

        # GLM Fit: Linear regression using a sliding window of 10 steps
        if t >= beta_window:
            X_exc = fixed_x[t-beta_window:t].reshape(-1, 1)  # Inputs, shape (beta_window, 1)
            Y_exc_window = y_exc_mod[:, t-beta_window:t]  # Outputs, shape (2, beta_window)
            for i in range(2):
                # Perform least squares regression for each neuron
                beta_exc[i, t], _, _, _ = np.linalg.lstsq(X_exc, Y_exc_window[i, :], rcond=None)
        else:
            if t > 0:
                beta_exc[:, t] = beta_exc[:, t-1]
            else:
                beta_exc[:, t] = np.zeros(2)

        # Residuals
        epsilon_exc[:, t] = y_exc_mod[:, t] - beta_exc[:, t] * fixed_x[t]

        # Drift in beta coefficients over the window
        if t >= beta_window:
            delta_beta_exc = beta_exc[:, t] - beta_exc[:, t-beta_window]
            beta_drift_exc.append(np.mean(np.abs(delta_beta_exc)))
        else:
            beta_drift_exc.append(0)

        # Noise correlation over the window
        if t >= beta_window:
            # Compute residuals over the window
            epsilon_window_exc = epsilon_exc[:, t-beta_window+1:t+1]  # Shape: (2, beta_window)
            # Compute pairwise correlations between residuals at consecutive time steps
            rho_epsilon_exc_list = []
            for w in range(beta_window - 1):
                if np.std(epsilon_window_exc[:, w]) > 1e-6 and np.std(epsilon_window_exc[:, w+1]) > 1e-6:
                    rho = np.corrcoef(epsilon_window_exc[:, w], epsilon_window_exc[:, w+1])[0, 1]
                else:
                    rho = 0
                rho_epsilon_exc_list.append(rho)
            # Average noise correlation over the window
            avg_rho_exc = np.mean(rho_epsilon_exc_list)
            noise_corr_exc.append(avg_rho_exc)
            # Change in noise correlation over the window
            if t >= 2 * beta_window:
                delta_rho_exc = abs(avg_rho_exc - noise_corr_exc[-2])
                noise_corr_change_exc.append(delta_rho_exc)
            else:
                noise_corr_change_exc.append(0)
        else:
            noise_corr_exc.append(0)
            noise_corr_change_exc.append(0)

        # Drift in outputs (Excitability Modulation Model) over the window
        if t >= beta_window:
            drift = np.var(y_exc_mod[:, t] - y_exc_mod[:, t-beta_window])
            drift_exc.append(drift)
            # Drift difference between neurons
            drift_diff = np.var((y_exc_mod[0, t] - y_exc_mod[0, t-beta_window]) - (y_exc_mod[1, t] - y_exc_mod[1, t-beta_window]))
            drift_diff_exc.append(drift_diff)
        else:
            drift_exc.append(0)
            drift_diff_exc.append(0)

        # Compute cost function for Excitability Modulation Model
        cost_current_exc = compute_cost(y_exc_mod[:, :t+1], fixed_x[:t+1])
        cost_exc.append(cost_current_exc)

    # ----- Post-Simulation Analysis -----

    # Convert lists to numpy arrays for easier manipulation
    drift_syn = np.array(drift_syn)
    drift_diff_syn = np.array(drift_diff_syn)
    drift_exc = np.array(drift_exc)
    drift_diff_exc = np.array(drift_diff_exc)
    beta_drift_syn = np.array(beta_drift_syn)
    beta_drift_exc = np.array(beta_drift_exc)
    noise_corr_syn = np.array(noise_corr_syn)
    noise_corr_exc = np.array(noise_corr_exc)
    noise_corr_change_syn = np.array(noise_corr_change_syn)
    noise_corr_change_exc = np.array(noise_corr_change_exc)
    cost_syn = np.array(cost_syn)
    cost_exc = np.array(cost_exc)

    # Compute mean values over time (excluding t < 2*beta_window for noise correlation changes)
    mean_drift_syn = np.mean(drift_syn[beta_window:])  # Exclude initial steps
    mean_drift_exc = np.mean(drift_exc[beta_window:])
    mean_drift_diff_syn = np.mean(drift_diff_syn[beta_window:])
    mean_drift_diff_exc = np.mean(drift_diff_exc[beta_window:])
    mean_beta_drift_syn = np.mean(beta_drift_syn[beta_window:])
    mean_beta_drift_exc = np.mean(beta_drift_exc[beta_window:])
    mean_noise_corr_syn = np.mean(noise_corr_syn[beta_window:])  # Exclude t < beta_window
    mean_noise_corr_exc = np.mean(noise_corr_exc[beta_window:])
    mean_change_noise_corr_syn = np.mean(noise_corr_change_syn[2*beta_window:])  # Exclude initial steps
    mean_change_noise_corr_exc = np.mean(noise_corr_change_exc[2*beta_window:])
    final_cost_syn = cost_syn[-1]
    final_cost_exc = cost_exc[-1]

    # Store results
    results['rho_zeta'].append(rho_zeta)
    results['mean_drift_syn'].append(mean_drift_syn)
    results['mean_drift_exc'].append(mean_drift_exc)
    results['mean_drift_diff_syn'].append(mean_drift_diff_syn)
    results['mean_drift_diff_exc'].append(mean_drift_diff_exc)
    results['mean_beta_drift_syn'].append(mean_beta_drift_syn)
    results['mean_beta_drift_exc'].append(mean_beta_drift_exc)
    results['mean_noise_corr_syn'].append(mean_noise_corr_syn)
    results['mean_noise_corr_exc'].append(mean_noise_corr_exc)
    results['mean_change_noise_corr_syn'].append(mean_change_noise_corr_syn)
    results['mean_change_noise_corr_exc'].append(mean_change_noise_corr_exc)
    results['final_cost_syn'].append(final_cost_syn)
    results['final_cost_exc'].append(final_cost_exc)

# ----------------------------
# Plotting Results
# ----------------------------

# Extract results for plotting
rho_zeta = np.array(results['rho_zeta'])
mean_drift_syn = np.array(results['mean_drift_syn'])
mean_drift_exc = np.array(results['mean_drift_exc'])
mean_drift_diff_syn = np.array(results['mean_drift_diff_syn'])
mean_drift_diff_exc = np.array(results['mean_drift_diff_exc'])
mean_beta_drift_syn = np.array(results['mean_beta_drift_syn'])
mean_beta_drift_exc = np.array(results['mean_beta_drift_exc'])
mean_noise_corr_syn = np.array(results['mean_noise_corr_syn'])
mean_noise_corr_exc = np.array(results['mean_noise_corr_exc'])
mean_change_noise_corr_syn = np.array(results['mean_change_noise_corr_syn'])
mean_change_noise_corr_exc = np.array(results['mean_change_noise_corr_exc'])
final_cost_syn = np.array(results['final_cost_syn'])
final_cost_exc = np.array(results['final_cost_exc'])

# Create subplots
fig, axs = plt.subplots(5, 2, figsize=(20, 25))

# 1. Noise Correlation vs Mean Drift
axs[0, 0].scatter(mean_noise_corr_syn, mean_drift_syn, color='blue', label='Synaptic Noise')
axs[0, 0].scatter(mean_noise_corr_exc, mean_drift_exc, color='orange', label='Excitability Modulation')
axs[0, 0].set_xlabel('Mean Noise Correlation ($\\rho_{\\epsilon}$)')
axs[0, 0].set_ylabel('Mean Drift (Variance)')
axs[0, 0].set_title('Noise Correlation vs Mean Drift')
axs[0, 0].legend()

# 2. Noise Correlation vs Drift Difference |drift_n1 - drift_n2|
axs[0, 1].scatter(mean_noise_corr_syn, mean_drift_diff_syn, color='blue', label='Synaptic Noise')
axs[0, 1].scatter(mean_noise_corr_exc, mean_drift_diff_exc, color='orange', label='Excitability Modulation')
axs[0, 1].set_xlabel('Mean Noise Correlation ($\\rho_{\\epsilon}$)')
axs[0, 1].set_ylabel('Drift Difference Variance ($\\text{Var}(|\\Delta y_1 - \\Delta y_2|)$)')
axs[0, 1].set_title('Noise Correlation vs Drift Difference Between Neurons')
axs[0, 1].legend()

# 3. Mean Drift vs Beta Drift
axs[1, 0].scatter(mean_beta_drift_syn, mean_drift_syn, color='blue', label='Synaptic Noise')
axs[1, 0].scatter(mean_beta_drift_exc, mean_drift_exc, color='orange', label='Excitability Modulation')
axs[1, 0].set_xlabel('Mean Beta Drift ($\\langle |\\Delta \\beta| \\rangle$)')
axs[1, 0].set_ylabel('Mean Drift (Variance)')
axs[1, 0].set_title('Beta Drift vs Mean Drift')
axs[1, 0].legend()

# 4. Change in Noise Correlation vs Mean Drift
axs[1, 1].scatter(mean_change_noise_corr_syn, mean_drift_syn, color='blue', label='Synaptic Noise')
axs[1, 1].scatter(mean_change_noise_corr_exc, mean_drift_exc, color='orange', label='Excitability Modulation')
axs[1, 1].set_xlabel('Mean Change in Noise Correlation ($\\langle |\\Delta \\rho_{\\epsilon}| \\rangle$)')
axs[1, 1].set_ylabel('Mean Drift (Variance)')
axs[1, 1].set_title('Change in Noise Correlation vs Mean Drift')
axs[1, 1].legend()

# 5. Excitability Correlation vs Noise Correlation (Excitability Modulation Model)
axs[2, 0].scatter(rho_zeta, mean_noise_corr_exc, color='orange', label='Excitability Modulation')
axs[2, 0].set_xlabel('Excitability Correlation ($\\rho_{\\zeta}$)')
axs[2, 0].set_ylabel('Mean Noise Correlation ($\\rho_{\\epsilon}$)')
axs[2, 0].set_title('Excitability Correlation vs Noise Correlation (Excitability Modulation Model)')
axs[2, 0].legend()

# 6. Cost Function Over Time for Synaptic Noise Model
# To visualize cost over time, we need to plot for one of the simulations
# However, since we have multiple rho_zeta values, we'll plot the final cost
axs[2, 1].plot(rho_zeta, final_cost_syn, marker='o', color='blue', label='Synaptic Noise')
axs[2, 1].set_xlabel('Excitability Correlation ($\\rho_{\\zeta}$)')
axs[2, 1].set_ylabel('Final Cost Function')
axs[2, 1].set_title('Final Cost Function vs Excitability Correlation (Synaptic Noise Model)')
axs[2, 1].legend()

# 7. Cost Function Over Time for Excitability Modulation Model
axs[3, 0].plot(rho_zeta, final_cost_exc, marker='o', color='orange', label='Excitability Modulation')
axs[3, 0].set_xlabel('Excitability Correlation ($\\rho_{\\zeta}$)')
axs[3, 0].set_ylabel('Final Cost Function')
axs[3, 0].set_title('Final Cost Function vs Excitability Correlation (Excitability Modulation Model)')
axs[3, 0].legend()

# 8. Change in Noise Correlation vs Drift Difference
axs[3, 1].scatter(mean_change_noise_corr_syn, mean_drift_diff_syn, color='blue', label='Synaptic Noise')
axs[3, 1].scatter(mean_change_noise_corr_exc, mean_drift_diff_exc, color='orange', label='Excitability Modulation')
axs[3, 1].set_xlabel('Mean Change in Noise Correlation ($\\langle |\\Delta \\rho_{\\epsilon}| \\rangle$)')
axs[3, 1].set_ylabel('Drift Difference Variance ($\\text{Var}(|\\Delta y_1 - \\Delta y_2|)$)')
axs[3, 1].set_title('Change in Noise Correlation vs Drift Difference')
axs[3, 1].legend()

# 9. Final Cost vs Excitability Correlation
axs[4, 0].scatter(rho_zeta, final_cost_syn, color='blue', label='Synaptic Noise')
axs[4, 0].scatter(rho_zeta, final_cost_exc, color='orange', label='Excitability Modulation')
axs[4, 0].set_xlabel('Excitability Correlation ($\\rho_{\\zeta}$)')
axs[4, 0].set_ylabel('Final Cost Function')
axs[4, 0].set_title('Final Cost Function vs Excitability Correlation')
axs[4, 0].legend()

# 10. Summary of Correlations
axs[4, 1].axis('off')  # Hide the subplot
axs[4, 1].text(0.0, 1.0, "Correlation Coefficients Summary:", fontsize=14, fontweight='bold', va='top')
for i, rho_zeta_val in enumerate(rho_zeta_values):
    text = f"\nrho_zeta = {rho_zeta_val}"
    text += f"\n  Synaptic Noise Model:"
    text += f"\n    Mean Noise Corr: {mean_noise_corr_syn[i]:.3f}"
    text += f"\n    Mean Drift: {mean_drift_syn[i]:.3f}"
    text += f"\n    Drift Difference: {mean_drift_diff_syn[i]:.3f}"
    text += f"\n    Mean Beta Drift: {mean_beta_drift_syn[i]:.3f}"
    text += f"\n    Final Cost: {final_cost_syn[i]:.3f}"
    text += f"\n  Excitability Modulation Model:"
    text += f"\n    Mean Noise Corr: {mean_noise_corr_exc[i]:.3f}"
    text += f"\n    Mean Drift: {mean_drift_exc[i]:.3f}"
    text += f"\n    Drift Difference: {mean_drift_diff_exc[i]:.3f}"
    text += f"\n    Mean Beta Drift: {mean_beta_drift_exc[i]:.3f}"
    text += f"\n    Final Cost: {final_cost_exc[i]:.3f}"
    axs[4, 1].text(0.0, 1.0 - (i+1)*0.2, text, fontsize=10, va='top')

plt.tight_layout()
plt.show()

# ----------------------------
# Correlation Coefficient Analysis
# ----------------------------

print("\n---- Summary of Correlation Coefficients ----\n")

for i, rho_zeta_val in enumerate(rho_zeta_values):
    print(f"rho_zeta = {rho_zeta_val}")
    print(f"Synaptic Noise Model:")
    print(f"  Mean Noise Correlation: {results['mean_noise_corr_syn'][i]:.3f}")
    print(f"  Mean Drift: {results['mean_drift_syn'][i]:.3f}")
    print(f"  Drift Difference: {results['mean_drift_diff_syn'][i]:.3f}")
    print(f"  Mean Beta Drift: {results['mean_beta_drift_syn'][i]:.3f}")
    print(f"  Final Cost Function: {results['final_cost_syn'][i]:.3f}")

    print(f"Excitability Modulation Model:")
    print(f"  Mean Noise Correlation: {results['mean_noise_corr_exc'][i]:.3f}")
    print(f"  Mean Drift: {results['mean_drift_exc'][i]:.3f}")
    print(f"  Drift Difference: {results['mean_drift_diff_exc'][i]:.3f}")
    print(f"  Mean Beta Drift: {results['mean_beta_drift_exc'][i]:.3f}")
    print(f"  Final Cost Function: {results['final_cost_exc'][i]:.3f}")
    print("-" * 50)