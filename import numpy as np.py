import numpy as np
import matplotlib.pyplot as plt

# Function to generate a correlated Gaussian distribution
def generate_correlated_gaussian(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)

# Generate random orthogonal vectors for eigenvectors
def generate_orthogonal_vectors(dim, n_vectors):
    random_matrix = np.random.randn(dim, n_vectors)
    q, _ = np.linalg.qr(random_matrix)
    return q

# Parameters
n = 100  # Input dimension
k = 3    # Output dimension (subspace)
T = 1000 # Number of iterations

# Covariance matrix
eigenvalues = np.array([4.5, 3.5, 1] + [0.01] * (n - 3))
Q = generate_orthogonal_vectors(n, n)
C = Q @ np.diag(eigenvalues) @ Q.T

# Learning parameters
η = 0.01  # Learning rate
σ = 0.01  # Noise standard deviation

# Initialization
W = np.random.randn(k, n)  # Forward connection matrix

# Record the output of the same input over time
y_record = np.zeros((k, T))

# Generate a single example input from the distribution
example_input = generate_correlated_gaussian(np.zeros(n), C, 1).T

for t in range(T):
    # Apply noisy update to W
    ξW = np.random.normal(0, η*σ, W.shape)
    W += η * (y_record[:, [t-1]] @ example_input.T - W) + ξW if t > 0 else 0
    
    # Project the example input through the updated W
    y_record[:, t] = W @ example_input[:, 0]

# Plotting
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(range(T), y_record[i, :], label=f'y{i+1}(t)')
plt.xlabel('Time (t)')
plt.ylabel('Output component')
plt.title('Drift in the Learned Representation Over Time')
plt.legend()
plt.show()
