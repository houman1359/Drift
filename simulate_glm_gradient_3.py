import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

# Simulation Parameters
eta = 0.01          # Learning rate
sigma_w = 0.1       # Feedforward weight noise standard deviation
sigma_m = 0.1       # Recurrent weight noise standard deviation
e = 0.5             # Excitability rate for Modulation Model
T = 1000            # Number of time steps
N_runs = 100        # Number of simulation runs for averaging

# Initialize Synaptic Weights (Diagonal for simplicity)
m = 1.0
M_initial = np.array([[m, 0.0],
                      [0.0, m]])

# Initialize Feedforward Weights
w_initial = np.array([1.0, 1.0])

# Compute Initial Beta Coefficients
beta_initial = inv(M_initial).dot(w_initial)

# Define R matrix with specified rho_noise
def get_R(rho_noise):
    return np.array([[1.0, rho_noise],
                     [rho_noise, 1.0]])

# Function to simulate Synaptic Noise Model
def simulate_synaptic_noise(rho_noise, T, N_runs):
    Var_D_pair_runs = []
    
    for run in range(N_runs):
        # Initialize weights for each run
        M = M_initial.copy()
        w = w_initial.copy()
        beta = inv(M).dot(w)
        
        # List to store drift differences for this run
        D_pair_list = []
        
        for t in range(T):
            # Generate input x_t (constant input)
            x_t = 1.0  
            
            # Compute output y_t
            y_t = beta * x_t
            
            # Update feedforward weights w
            Delta_w = eta * (y_t * x_t - w) + np.random.normal(0, sigma_w, size=w.shape)
            w += Delta_w
            
            # Update recurrent weights M
            # Generate noise for each row separately to match covariance dimensions
            R = get_R(rho_noise)
            noise_row1 = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_m**2 * R)
            noise_row2 = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_m**2 * R)
            noise = np.vstack([noise_row1, noise_row2])
            
            Delta_M = eta * (np.outer(y_t, y_t) - M) + noise
            M += Delta_M
            
            # Update beta
            try:
                beta_new = inv(M).dot(w)
            except np.linalg.LinAlgError:
                # In case M becomes singular, skip this update
                beta_new = beta  # No change if inversion fails
            
            # Compute drift difference D_pair = |Delta_beta1 - Delta_beta2|
            Delta_beta = beta_new - beta
            D_pair = np.abs(Delta_beta[0] - Delta_beta[1])
            D_pair_list.append(D_pair)
            
            # Update beta for next iteration
            beta = beta_new.copy()
        
        # Compute variance of D_pair for this run
        Var_D_pair = np.var(D_pair_list)
        Var_D_pair_runs.append(Var_D_pair)
    
    # Return list of variances for all runs
    return Var_D_pair_runs

# Function to simulate Excitability Modulation Model
def simulate_excitability_modulation(e, T, N_runs):
    Var_D_pair_mod_runs = []
    
    for run in range(N_runs):
        # Initialize weights for each run (constant)
        M = M_initial.copy()
        w = w_initial.copy()
        beta = inv(M).dot(w)
        
        # List to store drift differences for this run
        D_pair_mod_list = []
        
        for t in range(T):
            # Generate input x_t (constant input)
            x_t = 1.0  
            
            # Generate excitability factors zeta_i(t) ~ Bernoulli(e)
            zeta_t = np.random.binomial(1, e, size=2)
            zeta_t1 = np.random.binomial(1, e, size=2)
            
            # Compute drift for each neuron
            Drift_mod = beta * x_t * (zeta_t1 - zeta_t)
            
            # Compute absolute drift difference
            D_pair_mod = np.abs(Drift_mod[0] - Drift_mod[1])
            D_pair_mod_list.append(D_pair_mod)
        
        # Compute variance of D_pair_mod for this run
        Var_D_pair_mod = np.var(D_pair_mod_list)
        Var_D_pair_mod_runs.append(Var_D_pair_mod)
    
    # Return list of variances for all runs
    return Var_D_pair_mod_runs

# Parameters for Synaptic Noise Model
rho_noise_values = np.linspace(-0.9, 0.9, 19)  # From -0.9 to 0.9 in steps of 0.1

# Arrays to store variance results for Synaptic Noise Model
Var_D_pair_synaptic_all = []

print("Simulating Synaptic Noise Model...")
for rho in rho_noise_values:
    var_dp_runs = simulate_synaptic_noise(rho, T, N_runs)
    Var_D_pair_synaptic_all.append(var_dp_runs)
    print(f"rho_noise = {rho:.1f}, Var(D_pair) (mean ± std) = {np.mean(var_dp_runs):.4f} ± {np.std(var_dp_runs):.4f}")

# Simulate Excitability Modulation Model
print("\nSimulating Excitability Modulation Model...")
Var_D_pair_mod_all = simulate_excitability_modulation(e, T, N_runs)
Var_D_pair_mod_mean = np.mean(Var_D_pair_mod_all)
Var_D_pair_mod_std = np.std(Var_D_pair_mod_all)
print(f"Excitability Modulation Model: Var(D_pair_mod) (mean ± std) = {Var_D_pair_mod_mean:.4f} ± {Var_D_pair_mod_std:.4f}")

# Flatten Synaptic Noise Model data for correlation analysis
# Create arrays for all runs across all rho_noise_values
rho_noise_flat = []
Var_D_pair_flat = []

for idx, rho in enumerate(rho_noise_values):
    Var_D_pair_flat.extend(Var_D_pair_synaptic_all[idx])
    rho_noise_flat.extend([rho] * N_runs)

rho_noise_flat = np.array(rho_noise_flat)
Var_D_pair_flat = np.array(Var_D_pair_flat)

# Calculate Pearson correlation between rho_noise and Var(D_pair)
corr_synaptic, p_value_synaptic = pearsonr(rho_noise_flat, Var_D_pair_flat)
print(f"\nCorrelation between rho_noise and Var(D_pair) in Synaptic Noise Model: {corr_synaptic:.4f} (p-value: {p_value_synaptic:.4e})")

# Visualization: Variance of Drift Differences vs Noise Correlation
plt.figure(figsize=(12, 6))

# Plot Synaptic Noise Model results
for idx, rho in enumerate(rho_noise_values):
    plt.scatter([rho] * N_runs, Var_D_pair_synaptic_all[idx], alpha=0.5, label=f'rho={rho:.1f}' if idx == 0 else "")

# Plot Excitability Modulation Model as a horizontal line
plt.axhline(y=Var_D_pair_mod_mean, color='red', linestyle='-', label='Excitability Modulation Model (Simulation)')
plt.axhline(y=Var_D_pair_mod_mean + Var_D_pair_mod_std, color='red', linestyle='--', label='Excitability Modulation Model (±1 STD)')
plt.axhline(y=Var_D_pair_mod_mean - Var_D_pair_mod_std, color='red', linestyle='--')

plt.xlabel(r'Noise Correlation $\rho_{\text{noise}}$')
plt.ylabel(r'Variance of Drift Difference $\text{Var}(D_{\text{pair}})$')
plt.title('Variance of Drift Differences vs Noise Correlation')
plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Scatter Plot and Correlation between Drift and Noise Correlation
plt.figure(figsize=(8, 6))
plt.scatter(rho_noise_flat, Var_D_pair_flat, alpha=0.3, label='Synaptic Noise Model Runs')
plt.axhline(y=Var_D_pair_mod_mean, color='red', linestyle='-', label='Excitability Modulation Model (Mean)')
plt.xlabel(r'Noise Correlation $\rho_{\text{noise}}$')
plt.ylabel(r'Variance of Drift Difference $\text{Var}(D_{\text{pair}})$')
plt.title('Correlation between Noise Correlation and Variance of Drift Difference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Scatter Plot: Drift vs Delta Beta
# To analyze drift vs delta beta, we need to collect delta beta data
# However, in the current simulation setup, we only store Var(D_pair) per run
# To obtain delta beta, we need to modify the simulation to store Delta_beta per run

# Modify the simulate_synaptic_noise function to collect average |Delta_beta|
def simulate_synaptic_noise_with_delta_beta(rho_noise, T, N_runs):
    Var_D_pair_runs = []
    Avg_Delta_beta_runs = []
    
    for run in range(N_runs):
        # Initialize weights for each run
        M = M_initial.copy()
        w = w_initial.copy()
        beta = inv(M).dot(w)
        
        # Lists to store drift differences and delta betas for this run
        D_pair_list = []
        Delta_beta_list = []
        
        for t in range(T):
            # Generate input x_t (constant input)
            x_t = 1.0  
            
            # Compute output y_t
            y_t = beta * x_t
            
            # Update feedforward weights w
            Delta_w = eta * (y_t * x_t - w) + np.random.normal(0, sigma_w, size=w.shape)
            w += Delta_w
            
            # Update recurrent weights M
            # Generate noise for each row separately to match covariance dimensions
            R = get_R(rho_noise)
            noise_row1 = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_m**2 * R)
            noise_row2 = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_m**2 * R)
            noise = np.vstack([noise_row1, noise_row2])
            
            Delta_M = eta * (np.outer(y_t, y_t) - M) + noise
            M += Delta_M
            
            # Update beta
            try:
                beta_new = inv(M).dot(w)
            except np.linalg.LinAlgError:
                # In case M becomes singular, skip this update
                beta_new = beta  # No change if inversion fails
            
            # Compute drift difference D_pair = |Delta_beta1 - Delta_beta2|
            Delta_beta = beta_new - beta
            D_pair = np.abs(Delta_beta[0] - Delta_beta[1])
            D_pair_list.append(D_pair)
            
            # Compute average |Delta_beta|
            avg_Delta_beta = np.mean(np.abs(Delta_beta))
            Delta_beta_list.append(avg_Delta_beta)
            
            # Update beta for next iteration
            beta = beta_new.copy()
        
        # Compute variance of D_pair and average Delta_beta for this run
        Var_D_pair = np.var(D_pair_list)
        Avg_Delta_beta = np.mean(Delta_beta_list)
        Var_D_pair_runs.append(Var_D_pair)
        Avg_Delta_beta_runs.append(Avg_Delta_beta)
    
    # Return lists of variances and average Delta_betas for all runs
    return Var_D_pair_runs, Avg_Delta_beta_runs

# Re-simulate Synaptic Noise Model with Delta Beta Collection
print("\nSimulating Synaptic Noise Model with Delta Beta Collection...")
Var_D_pair_synaptic_all_with_delta, Avg_Delta_beta_synaptic_all = [], []
for rho in rho_noise_values:
    var_dp_runs, avg_db_runs = simulate_synaptic_noise_with_delta_beta(rho, T, N_runs)
    Var_D_pair_synaptic_all_with_delta.append(var_dp_runs)
    Avg_Delta_beta_synaptic_all.append(avg_db_runs)
    print(f"rho_noise = {rho:.1f}, Var(D_pair) (mean ± std) = {np.mean(var_dp_runs):.4f} ± {np.std(var_dp_runs):.4f}, "
          f"Avg |Delta_beta| (mean ± std) = {np.mean(avg_db_runs):.4f} ± {np.std(avg_db_runs):.4f}")

# Flatten data for correlation between Var(D_pair) and Avg |Delta_beta|
Var_D_pair_synaptic_flat = []
Avg_Delta_beta_flat = []
for var_list, db_list in zip(Var_D_pair_synaptic_all_with_delta, Avg_Delta_beta_synaptic_all):
    Var_D_pair_synaptic_flat.extend(var_list)
    Avg_Delta_beta_flat.extend(db_list)

Var_D_pair_synaptic_flat = np.array(Var_D_pair_synaptic_flat)
Avg_Delta_beta_flat = np.array(Avg_Delta_beta_flat)

# Calculate Pearson correlation between Var(D_pair) and Avg |Delta_beta|
corr_drift_delta_beta, p_value_drift_delta_beta = pearsonr(Var_D_pair_synaptic_flat, Avg_Delta_beta_flat)
print(f"\nCorrelation between Var(D_pair) and Avg |Delta_beta| in Synaptic Noise Model: {corr_drift_delta_beta:.4f} (p-value: {p_value_drift_delta_beta:.4e})")

# Visualization: Drift vs Delta Beta
plt.figure(figsize=(8, 6))
plt.scatter(Var_D_pair_synaptic_flat, Avg_Delta_beta_flat, alpha=0.3, label='Synaptic Noise Model Runs')
plt.xlabel(r'Variance of Drift Difference $\text{Var}(D_{\text{pair}})$')
plt.ylabel(r'Average $|\Delta \beta|$')
plt.title('Correlation between Drift Variance and Delta Beta in Synaptic Noise Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary of Correlations
print("\nSummary of Correlations:")
print(f"1. Correlation between rho_noise and Var(D_pair): {corr_synaptic:.4f} (p-value: {p_value_synaptic:.4e})")
print(f"2. Correlation between Var(D_pair) and Avg |Delta_beta|: {corr_drift_delta_beta:.4f} (p-value: {p_value_drift_delta_beta:.4e})")