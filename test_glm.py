import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Simulation parameters
n_simulations = 1000  # Number of simulations
timesteps = 100  # Number of timesteps
rho_epsilon_range = np.linspace(-1, 1, 100)  # Range of noise correlations to explore
noise_std = 0.05  # Increased standard deviation of noise for more drift
M = 0.5  # Increased recurrent weight for stronger interaction between neurons
W1 = 1.0  # Feedforward weight for neuron 1
W2 = 1.0  # Feedforward weight for neuron 2
input_strength = 2.0  # Increased input strength for larger variation

# Function to simulate the network with stronger noise and drift
def simulate_network(n_simulations, timesteps, rho_epsilon, noise_std, W1, W2, M, input_strength):
    drift_diff_results = []
    drift_single_results = []
    delta_glm_weights = []
    
    for _ in range(n_simulations):
        # Generate stronger input x_t
        x_t = input_strength * np.random.randn(timesteps)
        
        # Initialize output neurons
        y1 = np.zeros(timesteps)
        y2 = np.zeros(timesteps)
        
        # Noise with correlation rho_epsilon
        noise_epsilon_1 = np.random.randn(timesteps)
        noise_epsilon_2 = rho_epsilon * noise_epsilon_1 + np.sqrt(1 - rho_epsilon**2) * np.random.randn(timesteps)
        
        # Simulate the network dynamics
        for t in range(1, timesteps):
            y1[t] = W1 * x_t[t] - M * y2[t-1] + noise_std * noise_epsilon_1[t]
            y2[t] = W2 * x_t[t] - M * y1[t-1] + noise_std * noise_epsilon_2[t]

        # GLM Fit for both neurons
        model_1 = LinearRegression().fit(x_t.reshape(-1, 1), y1)
        model_2 = LinearRegression().fit(x_t.reshape(-1, 1), y2)
        beta1_t1 = model_1.coef_[0]
        beta2_t1 = model_2.coef_[0]
        
        # Recalculate for time t+1 (introducing more drift in GLM weights)
        beta1_t2 = beta1_t1 + np.random.randn() * noise_std * 2  # Increase variability in GLM weight updates
        beta2_t2 = beta2_t1 + np.random.randn() * noise_std * 2
        
        # Store delta GLM weights
        delta_glm_weights.append([beta1_t2 - beta1_t1, beta2_t2 - beta2_t1])
        
        # Calculate drift for y1 and y2
        drift_y1 = np.abs(beta1_t2 * x_t - beta1_t1 * x_t).mean()
        drift_y2 = np.abs(beta2_t2 * x_t - beta2_t1 * x_t).mean()
        
        # Store results
        drift_single_results.append([drift_y1, drift_y2])
        drift_diff_results.append(np.abs(drift_y1 - drift_y2))
        
    return drift_single_results, drift_diff_results, delta_glm_weights

# Perform simulations across different noise correlations
all_drift_single = []
all_drift_diff = []
all_delta_glm_weights = []
all_rho_epsilon = []

for rho_epsilon in rho_epsilon_range:
    drift_single, drift_diff, delta_glm_weight = simulate_network(n_simulations, timesteps, rho_epsilon, noise_std, W1, W2, M, input_strength)
    all_drift_single.extend(drift_single)
    all_drift_diff.extend(drift_diff)
    all_delta_glm_weights.extend(delta_glm_weight)
    all_rho_epsilon.extend([rho_epsilon] * n_simulations)

# Convert to arrays
all_drift_single = np.array(all_drift_single)
all_delta_glm_weights = np.array(all_delta_glm_weights)
all_rho_epsilon = np.array(all_rho_epsilon)

# Scatter plot for Drift y1, y2, and Drift Difference against Noise Correlation
plt.figure(figsize=(10, 6))
plt.scatter(all_rho_epsilon, all_drift_single[:, 0], label='Drift y1', alpha=0.5)
plt.scatter(all_rho_epsilon, all_drift_single[:, 1], label='Drift y2', alpha=0.5)
plt.scatter(all_rho_epsilon, all_drift_diff, label='Drift Difference', alpha=0.5)
plt.xlabel('Noise Correlation (rho_epsilon)')
plt.ylabel('Drift')
plt.legend()
plt.title('Scatter Plot: Drift vs. Noise Correlation')
plt.show()

# Scatter plot for Delta GLM Weights against Noise Correlation
plt.figure(figsize=(10, 6))
plt.scatter(all_rho_epsilon, all_delta_glm_weights[:, 0], label='Delta GLM Weight y1', alpha=0.5)
plt.scatter(all_rho_epsilon, all_delta_glm_weights[:, 1], label='Delta GLM Weight y2', alpha=0.5)
plt.xlabel('Noise Correlation (rho_epsilon)')
plt.ylabel('Delta GLM Weights')
plt.legend()
plt.title('Scatter Plot: Delta GLM Weights vs. Noise Correlation')
plt.show()

# Compute Pearson correlation coefficients
corr_drift_y1, _ = pearsonr(all_rho_epsilon, all_drift_single[:, 0])
corr_drift_y2, _ = pearsonr(all_rho_epsilon, all_drift_single[:, 1])
corr_drift_diff, _ = pearsonr(all_rho_epsilon, all_drift_diff)
corr_delta_glm_y1, _ = pearsonr(all_rho_epsilon, all_delta_glm_weights[:, 0])
corr_delta_glm_y2, _ = pearsonr(all_rho_epsilon, all_delta_glm_weights[:, 1])

# Print correlation results
print(f'Correlation between Noise Correlation and Drift y1: {corr_drift_y1:.3f}')
print(f'Correlation between Noise Correlation and Drift y2: {corr_drift_y2:.3f}')
print(f'Correlation between Noise Correlation and Drift Difference: {corr_drift_diff:.3f}')
print(f'Correlation between Noise Correlation and Delta GLM Weight y1: {corr_delta_glm_y1:.3f}')
print(f'Correlation between Noise Correlation and Delta GLM Weight y2: {corr_delta_glm_y2:.3f}')

# # Set random seed for reproducibility
# np.random.seed(42)

# # Simulation parameters
# T = 5  # Number of time steps (samples)
# n = 1  # Dimensionality of input
# k = 2  # Number of output neurons
# sigma_w = 0.01  # Standard deviation of weight noise
# sigma_m = 0.01  # Standard deviation of recurrent noise
# eta = 0.01  # Learning rate
# num_simulations = 5000  # Number of simulations

# # Initialize input and weight matrices
# x = np.random.randn(T, n)  # Input (one-dimensional)
# W = np.random.randn(k, n)  # Feedforward weights
# M = np.random.randn(k, k)  # Recurrent weights
# M = M - np.diag(np.diag(M))  # Remove self-connections

# # Arrays to store results
# drift_1_vals = []
# drift_2_vals = []
# drift_diff_vals = []
# noise_corr_vals = []

# # Simulate network with noisy weight updates and track drifts and correlations
# for sim in range(num_simulations):
#     # Initialize output and noise matrices
#     y = np.zeros((T, k))  # Outputs
    
#     # Simulate network dynamics
#     for t in range(1, T):
#         y[t] = W @ x[t] + M @ y[t - 1] + np.random.randn(k) * sigma_w
    
#     # Perform GLM (Linear Regression)
#     glm_1 = LinearRegression().fit(x, y[:, 0])
#     glm_2 = LinearRegression().fit(x, y[:, 1])

#     # Compute residuals
#     residuals_1 = y[:, 0] - glm_1.predict(x)
#     residuals_2 = y[:, 1] - glm_2.predict(x)

#     # Compute original noise correlation (from GLM residuals)
#     noise_corr_orig = np.corrcoef(residuals_1, residuals_2)[0, 1]

#     # Simulate small weight updates
#     delta_W_1 = np.random.randn(n) * sigma_w
#     delta_W_2 = np.random.randn(n) * sigma_w
#     delta_M_12 = np.random.randn() * sigma_m
#     delta_M_21 = np.random.randn() * sigma_m

#     # Update residuals after weight shift
#     residuals_1_new = residuals_1 - delta_W_1 * x[:, 0]
#     residuals_2_new = residuals_2 - delta_W_2 * x[:, 0]

#     # Compute new noise correlation
#     noise_corr_new = np.corrcoef(residuals_1_new, residuals_2_new)[0, 1]

#     # Compute drift changes
#     drift_1 = np.var(delta_W_1 * x[:, 0] + (M[0, 1] + delta_M_12) * y[:, 1])
#     drift_2 = np.var(delta_W_2 * x[:, 0] + (M[1, 0] + delta_M_21) * y[:, 0])
#     drift_diff = drift_1 - drift_2

#     # Store results
#     drift_1_vals.append(drift_1)
#     drift_2_vals.append(drift_2)
#     drift_diff_vals.append(drift_diff)
#     noise_corr_vals.append(noise_corr_new)

# # Plot drift difference vs. noise correlation
# plt.figure(figsize=(10, 6))
# plt.scatter(noise_corr_vals, drift_diff_vals, color='blue', label="Drift Difference")
# plt.title("Drift Difference vs. Noise Correlation (Multiple Simulations)")
# plt.xlabel("Noise Correlation (GLM Residuals)")
# plt.ylabel("Drift Difference (Var(Δy1) - Var(Δy2))")
# plt.grid(True)
# plt.legend()
# plt.show()

# # Plot individual drifts vs. noise correlation
# plt.figure(figsize=(10, 6))
# plt.scatter(noise_corr_vals, drift_1_vals, color='red', label="Drift of Output 1")
# plt.scatter(noise_corr_vals, drift_2_vals, color='green', label="Drift of Output 2")
# plt.title("Drifts vs. Noise Correlation (Multiple Simulations)")
# plt.xlabel("Noise Correlation (GLM Residuals)")
# plt.ylabel("Drift")
# plt.grid(True)
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# num_simulations = 10000
# num_samples = 10
# x = 1.0  # fixed input

# # Noise variance (same for both neurons)
# sigma_epsilon = 0.01

# # Function to simulate drift and noise correlation
# def simulate_drift_and_noise_correlation(rho_epsilon):
#     # Generate random weight updates for W1 and W2
#     delta_W1 = np.random.randn(num_samples)
#     delta_W2 = np.random.randn(num_samples)
    
#     # Generate correlated noise between epsilon1 and epsilon2
#     cov_matrix = np.array([[1, rho_epsilon], [rho_epsilon, 1]]) * sigma_epsilon**2
#     noise = np.random.multivariate_normal([0, 0], cov_matrix, size=num_samples)
    
#     epsilon1, epsilon2 = noise[:, 0], noise[:, 1]
    
#     # Compute output variances
#     var_y1 = x**2 * np.var(delta_W1) + np.var(epsilon1)
#     var_y2 = x**2 * np.var(delta_W2) + np.var(epsilon2)
    
#     # Compute drift difference
#     drift_difference = x**2 * (np.var(delta_W1) - np.var(delta_W2))
    
#     return drift_difference, np.corrcoef(epsilon1, epsilon2)[0, 1]

# # Simulation
# rho_epsilon_values = np.linspace(0, 1, num_simulations)
# drift_differences = []
# noise_correlations = []

# for rho_epsilon in rho_epsilon_values:
#     drift_diff, noise_corr = simulate_drift_and_noise_correlation(rho_epsilon)
#     drift_differences.append(drift_diff)
#     noise_correlations.append(noise_corr)

# # Plot results
# plt.figure(figsize=(10, 6))
# plt.plot(noise_correlations, drift_differences, 'o', label='Drift Difference vs Noise Correlation')
# plt.xlabel('Noise Correlation')
# plt.ylabel('Drift Difference (Var(Δy1) - Var(Δy2))')
# plt.title('Drift Difference vs Noise Correlation')
# plt.legend()
# plt.grid(True)
# plt.show()
