import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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
    mean = np.zeros(n_neurons)
    cov = np.full((n_neurons, n_neurons), rho)
    np.fill_diagonal(cov, 1)

    latent = multivariate_normal.rvs(mean=mean, cov=cov, size=T)
    if n_neurons == 1:
        latent = latent.reshape(-1, 1)

    threshold = np.percentile(latent, 100 * (1 - e), axis=0)
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
eta = 0.01  # Hebbian learning rate
gamma = 0.001  # Correction learning rate for gradient descent
e = 0.5  # Excitability rate
sigma_w = 0.05  # Synaptic noise std for feedforward weights
sigma_m = 0.05  # Synaptic noise std for recurrent weights
beta_window = 10  # Window size for GLM fitting
rho_zeta_values = [0.0, 0.2, 0.4, 0.6, 0.8]  # Different excitability correlations
N_runs = 100  # Number of simulations per rho_zeta

# Initialize input signal (fixed for drift computation)
np.random.seed(44)  # For reproducibility
x_t = np.random.normal(0, 1, T)
fixed_x = x_t.copy()  # Keeping x_t fixed as per user instruction

# Initialize lists to store simulation results
# Using lists of lists to store results for multiple runs
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
    print(f"Running simulations for rho_zeta = {rho_zeta}")
    
    # Temporary lists to accumulate results for current rho_zeta
    temp_mean_drift_syn = []
    temp_mean_drift_exc = []
    temp_mean_drift_diff_syn = []
    temp_mean_drift_diff_exc = []
    temp_mean_beta_drift_syn = []
    temp_mean_beta_drift_exc = []
    temp_mean_noise_corr_syn = []
    temp_mean_noise_corr_exc = []
    temp_mean_change_noise_corr_syn = []
    temp_mean_change_noise_corr_exc = []
    temp_final_cost_syn = []
    temp_final_cost_exc = []
    
    for run in range(N_runs):
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
                X_syn = fixed_x[t-beta_window:t].reshape(-1, 1)  # Inputs, shape (10, 1)
                Y_syn_window = y_syn[:, t-beta_window:t]         # Outputs, shape (2, 10)
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
                epsilon_window_syn = epsilon_syn[:, t-beta_window+1:t+1]  # Shape: (2, 10)
                # Compute pairwise correlations between residuals at consecutive time steps
                rho_epsilon_syn_list = []
                for w_idx in range(beta_window - 1):
                    if np.std(epsilon_window_syn[:, w_idx]) > 1e-6 and np.std(epsilon_window_syn[:, w_idx+1]) > 1e-6:
                        rho = np.corrcoef(epsilon_window_syn[:, w_idx], epsilon_window_syn[:, w_idx+1])[0, 1]
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

            # ----- Gradient-Based Correction Step -----
            if t >= beta_window:
                # Compute the gradient of the cost with respect to w_syn
                y_window_syn = y_syn[:, t-beta_window:t]  # Shape: (2,10)
                x_window = fixed_x[t-beta_window:t]      # Shape: (10,)

                # Compute similarity matrices
                S_x = np.outer(x_window, x_window)       # Shape: (10,10)
                S_y = y_window_syn.T @ y_window_syn      # Shape: (10,10)

                diff = S_x - S_y                          # Shape: (10,10)

                # Compute gradient dL/dy correctly
                dL_dy = -2 * (y_window_syn @ diff)        # Shape: (2,10)

                # Compute gradient dL_dw_syn
                grad = dL_dy * x_window.reshape(1, -1)   # Shape: (2,10)
                sum_grad = grad.sum(axis=1)               # Shape: (2,)
                inv_M_syn_T = np.linalg.inv(M_syn).T      # Shape: (2,2)
                dL_dw_syn = inv_M_syn_T @ sum_grad        # Shape: (2,)

                # Update weights to minimize cost
                w_syn -= gamma * dL_dw_syn

            # Apply Hebbian and noise updates
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
                X_exc = fixed_x[t-beta_window:t].reshape(-1, 1)  # Inputs, shape (10, 1)
                Y_exc_window = y_exc_mod[:, t-beta_window:t]     # Outputs, shape (2, 10)
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
                epsilon_window_exc = epsilon_exc[:, t-beta_window+1:t+1]  # Shape: (2, 10)
                # Compute pairwise correlations between residuals at consecutive time steps
                rho_epsilon_exc_list = []
                for w_idx in range(beta_window - 1):
                    if np.std(epsilon_window_exc[:, w_idx]) > 1e-6 and np.std(epsilon_window_exc[:, w_idx+1]) > 1e-6:
                        rho = np.corrcoef(epsilon_window_exc[:, w_idx], epsilon_window_exc[:, w_idx+1])[0, 1]
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

        # Accumulate results from current run
        temp_mean_drift_syn.append(mean_drift_syn)
        temp_mean_drift_exc.append(mean_drift_exc)
        temp_mean_drift_diff_syn.append(mean_drift_diff_syn)
        temp_mean_drift_diff_exc.append(mean_drift_diff_exc)
        temp_mean_beta_drift_syn.append(mean_beta_drift_syn)
        temp_mean_beta_drift_exc.append(mean_beta_drift_exc)
        temp_mean_noise_corr_syn.append(mean_noise_corr_syn)
        temp_mean_noise_corr_exc.append(mean_noise_corr_exc)
        temp_mean_change_noise_corr_syn.append(mean_change_noise_corr_syn)
        temp_mean_change_noise_corr_exc.append(mean_change_noise_corr_exc)
        temp_final_cost_syn.append(final_cost_syn)
        temp_final_cost_exc.append(final_cost_exc)

    # After all runs for current rho_zeta, append to the main results
    results['rho_zeta'].extend([rho_zeta]*N_runs)
    results['mean_drift_syn'].extend(temp_mean_drift_syn)
    results['mean_drift_exc'].extend(temp_mean_drift_exc)
    results['mean_drift_diff_syn'].extend(temp_mean_drift_diff_syn)
    results['mean_drift_diff_exc'].extend(temp_mean_drift_diff_exc)
    results['mean_beta_drift_syn'].extend(temp_mean_beta_drift_syn)
    results['mean_beta_drift_exc'].extend(temp_mean_beta_drift_exc)
    results['mean_noise_corr_syn'].extend(temp_mean_noise_corr_syn)
    results['mean_noise_corr_exc'].extend(temp_mean_noise_corr_exc)
    results['mean_change_noise_corr_syn'].extend(temp_mean_change_noise_corr_syn)
    results['mean_change_noise_corr_exc'].extend(temp_mean_change_noise_corr_exc)
    results['final_cost_syn'].extend(temp_final_cost_syn)
    results['final_cost_exc'].extend(temp_final_cost_exc)

# ----------------------------
# Post-Simulation Correlation Analysis
# ----------------------------

# Convert results to numpy arrays for easier manipulation
rho_zeta_all = np.array(results['rho_zeta'])
mean_drift_syn_all = np.array(results['mean_drift_syn'])
mean_drift_exc_all = np.array(results['mean_drift_exc'])
mean_drift_diff_syn_all = np.array(results['mean_drift_diff_syn'])
mean_drift_diff_exc_all = np.array(results['mean_drift_diff_exc'])
mean_beta_drift_syn_all = np.array(results['mean_beta_drift_syn'])
mean_beta_drift_exc_all = np.array(results['mean_beta_drift_exc'])
mean_noise_corr_syn_all = np.array(results['mean_noise_corr_syn'])
mean_noise_corr_exc_all = np.array(results['mean_noise_corr_exc'])
mean_change_noise_corr_syn_all = np.array(results['mean_change_noise_corr_syn'])
mean_change_noise_corr_exc_all = np.array(results['mean_change_noise_corr_exc'])
final_cost_syn_all = np.array(results['final_cost_syn'])
final_cost_exc_all = np.array(results['final_cost_exc'])

# Define pairs for scatter plots and their labels
scatter_pairs = [
    ('mean_noise_corr_syn', 'mean_drift_syn', 'Mean Noise Corr (Synaptic) vs Mean Drift (Synaptic)'),
    ('mean_noise_corr_exc', 'mean_drift_exc', 'Mean Noise Corr (Excitability) vs Mean Drift (Excitability)'),
    ('mean_noise_corr_syn', 'mean_drift_diff_syn', 'Mean Noise Corr (Synaptic) vs Drift Difference (Synaptic)'),
    ('mean_noise_corr_exc', 'mean_drift_diff_exc', 'Mean Noise Corr (Excitability) vs Drift Difference (Excitability)'),
    ('mean_beta_drift_syn', 'mean_drift_syn', 'Mean Beta Drift (Synaptic) vs Mean Drift (Synaptic)'),
    ('mean_beta_drift_exc', 'mean_drift_exc', 'Mean Beta Drift (Excitability) vs Mean Drift (Excitability)'),
    ('mean_change_noise_corr_syn', 'mean_drift_syn', 'Mean Change Noise Corr (Synaptic) vs Mean Drift (Synaptic)'),
    ('mean_change_noise_corr_exc', 'mean_drift_exc', 'Mean Change Noise Corr (Excitability) vs Mean Drift (Excitability)'),
    ('rho_zeta', 'final_cost_syn', 'rho_zeta vs Final Cost (Synaptic)'),
    ('rho_zeta', 'final_cost_exc', 'rho_zeta vs Final Cost (Excitability)')
]

# ----------------------------
# Plotting Results with Correlation Coefficients
# ----------------------------

fig, axs = plt.subplots(5, 2, figsize=(25, 30))
fig.suptitle('Simulation Results with Correlation Coefficients', fontsize=16)

for idx, (x_label, y_label, title) in enumerate(scatter_pairs):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]
    
    # Select data based on labels
    x_data = locals()[f"{x_label}_all"]
    y_data = locals()[f"{y_label}_all"]
    
    # Compute Pearson correlation coefficient
    corr_coef, p_value = pearsonr(x_data, y_data)
    
    # Create scatter plot
    ax.scatter(x_data, y_data, alpha=0.5, edgecolor='k', linewidth=0.5)
    ax.set_xlabel(x_label.replace('_', ' ').capitalize())
    ax.set_ylabel(y_label.replace('_', ' ').capitalize())
    ax.set_title(f"{title}\nPearson r = {corr_coef:.2f}, p = {p_value:.2e}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ----------------------------
# Optional: Detailed Summary Table
# ----------------------------

import pandas as pd

# Create a DataFrame for easier analysis
df = pd.DataFrame({
    'rho_zeta': rho_zeta_all,
    'mean_drift_syn': mean_drift_syn_all,
    'mean_drift_exc': mean_drift_exc_all,
    'mean_drift_diff_syn': mean_drift_diff_syn_all,
    'mean_drift_diff_exc': mean_drift_diff_exc_all,
    'mean_beta_drift_syn': mean_beta_drift_syn_all,
    'mean_beta_drift_exc': mean_beta_drift_exc_all,
    'mean_noise_corr_syn': mean_noise_corr_syn_all,
    'mean_noise_corr_exc': mean_noise_corr_exc_all,
    'mean_change_noise_corr_syn': mean_change_noise_corr_syn_all,
    'mean_change_noise_corr_exc': mean_change_noise_corr_exc_all,
    'final_cost_syn': final_cost_syn_all,
    'final_cost_exc': final_cost_exc_all
})

# Display the first few rows of the DataFrame
print("\n---- First 5 Rows of the Simulation Data ----\n")
print(df.head())

# Compute overall correlation matrix
correlation_matrix = df.corr()
print("\n---- Correlation Matrix ----\n")
print(correlation_matrix)