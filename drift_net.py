import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
import numpy as np
from scipy.optimize import curve_fit
import sys
from scipy.spatial import ConvexHull


##############################################################################
##############################################################################

# class NoisySGD(optim.Optimizer):
#     """
#     Implements stochastic gradient descent (optionally with momentum) with additional noise,
#     making it consistent with manual updates when auto=0.
#     """
#     def __init__(self, params, lr, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False, noise_std=0.01):
#         if lr <= 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
#         defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, nesterov=nesterov, noise_std=noise_std)
#         super(NoisySGD, self).__init__(params, defaults)

#     def step(self, closure=None):
#         """
#         Performs a single optimization step, including noise addition consistent with auto=0 scenario.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 if group['weight_decay'] != 0:
#                     d_p.add_(p.data, alpha=group['weight_decay'])
#                 if group['momentum'] != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['dampening'])
#                     if group['nesterov']:
#                         d_p = d_p.add(buf, alpha=group['momentum'])
#                     else:
#                         d_p = buf

#                 # Adjust noise application to be consistent with auto=0 part
#                 noise = torch.sqrt(torch.tensor(group['lr'])) * torch.randn_like(p.data) * group['noise_std']
#                 p.data.add_(-group['lr'], d_p).add_(noise)  # Apply update and noise

#         return loss


def estimate_volume_convex_hull(points):
    """
    Estimate the volume occupied by a set of points in n-dimensional space
    by computing the volume of their convex hull.

    Parameters:
    - points: A numpy array of shape (num_points, n_dimensions) representing
              the points in n-dimensional space.

    Returns:
    - volume: The volume of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except scipy.spatial.qhull.QhullError:
        print("Error computing the convex hull. The points may be collinear or not span the entire space.")
        return None



##############################################################################
##############################################################################

def generate_PSP_input_torch(input_cov_eigens, input_dim, num_samples):
    Q, _ = torch.linalg.qr(torch.randn(input_dim, input_dim))
    C = Q @ torch.diag(torch.tensor(input_cov_eigens)) @ Q.T
    mean = torch.zeros(input_dim)
    X = torch.distributions.MultivariateNormal(mean, C).sample((num_samples,))
    return X

##############################################################################
##############################################################################

class SimilarityMatchingNetwork_WM(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimilarityMatchingNetwork_WM, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(output_dim, input_dim))
        self.M = torch.nn.Parameter(torch.eye(output_dim))
        
    def forward(self, x):
        # Compute output y = M^-1 * W * x
        # Create a temporary modified M with noise
        M_noisy = self.M #+ torch.randn_like(self.M) * noise
        W_noisy = self.W #+ torch.randn_like(self.W) * noise
        I = torch.eye(M_noisy.size(0))

        det_A = torch.det(M_noisy)
        is_singular = torch.isclose(det_A, torch.tensor(0.0, device=det_A.device))

        if not is_singular:
            M_inv = torch.inverse(M_noisy)
        else:
            M_inv = torch.full(M_noisy.shape, float('nan'), device=M_noisy.device)

        # Apply transformation using temporary noisy M
        y = M_inv @ W_noisy @ x.t()
        return y.t()

##############################################################################
##############################################################################

# class SimilarityMatchingNetwork_wCw(nn.Module):
#     def __init__(self, input_dim, output_dim, C):
#         super(SimilarityMatchingNetwork_wCw, self).__init__()
#         self.W = nn.Parameter(torch.randn(output_dim, input_dim))  # This remains for the linear transformation
#         self.w = nn.Parameter(torch.ones(output_dim))  # Diagonal elements of W for M = W C W^T
#         self.register_buffer('C', C)  # Register C as a constant buffer, not a learnable parameter

#     def forward(self, x):
#         # Construct diagonal matrix W from self.w
#         W_noisy = self.W #+ torch.randn_like(self.W) * noise
#         W_diag = torch.diag(self.w)
#         # Compute M using W C W^T
#         M = W_diag @ self.C @ W_diag 
#         #M += torch.randn_like(W_diag) * noise
#         I = torch.eye(M.size(0))
#         det_A = torch.det(M)
#         is_singular = torch.isclose(det_A, torch.tensor(0.0, device=det_A.device))
#         if not is_singular:
#             M_inv = torch.inverse(M)
#         else:
#             M_inv = torch.full(M.shape , 1.0, device=M.device)
#         # Apply transformation
#         y = M_inv @ W_noisy @ x.t()
#         return y.t()
    
##############################################################################
##############################################################################

def similarity_matching_cost(x, y):
    T = x.shape[0]  # Number of samples
    S_x = torch.matmul(x, x.t())
    S_y = torch.matmul(y, y.t())
    cost = torch.mean((S_x - S_y) ** 2) / (T ** 2)
    return cost

############################################################################## 
##############################################################################
#############################  MAIN PART  ####################################
##############################################################################
##############################################################################

# Generate input data
input_dim = 10  # Example dimensions
output_dim = 3
num_samples = 10000
tot_iter = 100000
syn_noise_std = 0.2
learnRate = 0.1
dt = learnRate
stdW = syn_noise_std
stdM = syn_noise_std
num_sel = 200 # randomly selected samples used to calculate the drift and diffusion constants
step = 10    # store every 10 updates
time_points = round(tot_iter / step)
sel_inx = np.random.permutation(num_samples)[:num_sel]

Yt_WM = np.zeros((output_dim, time_points, num_sel))
Yt_wCw = np.zeros((output_dim, time_points, num_sel))

eigenvalues = [4.5, 3.5, 1]  + [0.01] * (input_dim - 3)  # Eigenvalues for the covariance matrix
#eigenvalues = [0.1, 0.1, 0.1]   + [0.01] * (input_dim - 3)  # Eigenvalues for the covariance matrix
X = generate_PSP_input_torch(eigenvalues, input_dim, num_samples)

rho = 0.3 
C_target_np = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])  # Target correlation matrix
#C_target_np = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Target correlation matrix
C_target = torch.tensor(C_target_np, dtype=torch.float32)
# Multiply non-diagonal elements by -1
#C_target_mod = C_target.clone()  # Clone to avoid modifying the original tensor
C_target[~torch.eye(3, dtype=bool)] *= -1  # Only modify non-diagonal elements

model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim)
model_wCw = SimilarityMatchingNetwork_wCw(input_dim, output_dim, C_target)

#optimizer_WM = NoisySGD(model_WM.parameters(), lr=learnRate, noise_std=syn_noise_std)
#optimizer_wCw = NoisySGD(model_wCw.parameters(), lr=learnRate, noise_std=syn_noise_std)
optimizer_WM = torch.optim.SGD(model_WM.parameters(), lr=learnRate)
optimizer_wCw = torch.optim.SGD(model_wCw.parameters(), lr=learnRate)

DeltaWM_W_manual = torch.nn.Parameter(torch.randn(output_dim, input_dim))
DeltaWM_M_manual = torch.nn.Parameter(torch.eye(output_dim))
DeltawCw_W_manual = torch.nn.Parameter(torch.randn(output_dim, input_dim))
DeltawCw_w_manual = torch.nn.Parameter(torch.eye(output_dim))
nn = 0

for epoch in range(tot_iter):  # Number of epochs

    # Randomly select one sample
    curr_inx = torch.randint(0, num_samples, (1,))
    x_curr = X[curr_inx,:]  # Current input sample
    Y_WM = model_WM(x_curr)
    Y_wCw = model_wCw(x_curr)

    auto = 0

    optimizer_WM.zero_grad()
    optimizer_wCw.zero_grad()
    # Automatic gradients
    cost_WM = similarity_matching_cost(x_curr, Y_WM)
    cost_WM.backward()
    cost_wCw = similarity_matching_cost(x_curr, Y_wCw)
    cost_wCw.backward()

    if auto == 1:
        autoWM_grad_W = model_WM.W.grad.clone()
        autoWM_grad_M = model_WM.M.grad.clone()
        autowCw_grad_W = model_wCw.W.grad.clone()
        autowCw_grad_w = model_wCw.w.grad.clone()
        optimizer_WM.step()
        optimizer_wCw.step()

    # Generate noise matrices
    xis = torch.randn(output_dim, input_dim) * stdW 
    zetas = torch.randn(output_dim, output_dim) * stdM 

    # Update W and M with noise
    DeltaWM_W_manual = dt * (torch.matmul(Y_WM.t(), x_curr) / x_curr.size(0) - model_WM.W)+ torch.sqrt(torch.tensor(dt)) * xis  # y_i x_j - W_ij
    #DeltaWM_M_manual = dt * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model_WM.M)+ torch.sqrt(torch.tensor(dt)) * zetas  # y_i y_j - M_ij torch.sqrt(torch.tensor(dt))
    
    M = model_WM.M
    C = C_target
    M_diag = torch.diag(M)
    M_ii = M_diag.unsqueeze(1)  # Make it a column vector
    M_jj = M_diag.unsqueeze(0)  # Make it a row vector
    # Calculate E using broadcasting, avoiding division by zero or invalid operations
    E = torch.zeros_like(M)
    # Mask to avoid division for i=j (diagonal elements)
    mask = torch.eye(M.size(0), dtype=torch.bool)
    # Perform element-wise operation for i != j
    E[~mask] = (M[~mask] / (M_ii * M_jj)[~mask]) - (C[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask])
    # E now contains the calculated values based on the given conditions, with zeros on the diagonal

    DeltaWM_M_manual = dt * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model_WM.M- E)+ torch.sqrt(torch.tensor(dt)) * zetas  # y_i y_j - M_ij torch.sqrt(torch.tensor(dt))

    # DeltawCw_W_manual = dt * (torch.matmul(Y_wCw.t(), x_curr) / x_curr.size(0) - model_wCw.W)+ torch.sqrt(torch.tensor(dt)) * xis  # y_i x_j - W_ij
    # DeltawCw_M_manual =  (torch.matmul(Y_wCw.t(), Y_wCw) / Y_wCw.size(0) - torch.diag(model_wCw.w) @ model_wCw.C @ torch.diag(model_wCw.w)) #+  zetas  # y_i y_j - M_ij

    # dL_dM = DeltawCw_M_manual
    # dL_dw = torch.zeros_like(model_wCw.w)
    # for i in range(model_wCw.w.size(0)):
    #     for j in range(model_wCw.w.size(0)):
    #         dL_dw[i] += dL_dM[i, j] * model_wCw.w[j] * model_wCw.C[i, j] + dL_dM[j, i] * model_wCw.w[j] * model_wCw.C[j, i]
    # noise_w = torch.randn_like(model_wCw.w) *  (stdM) 
    # DeltawCw_w_manual =   torch.sqrt(torch.tensor(dt)) * dL_dw/10  + torch.sqrt(torch.tensor(dt)) * noise_w        
 
    if auto == 0:
        if torch.all(torch.isfinite(DeltaWM_W_manual)) and torch.all(torch.isfinite(DeltaWM_M_manual)):
            model_WM.W.data += DeltaWM_W_manual
            model_WM.M.data += DeltaWM_M_manual
            # Y_WM = model_WM(x_curr)
            # cost_WM = similarity_matching_cost(x_curr, Y_WM)
            # if torch.any(torch.isnan(cost_WM)):
            #     model_WM.W.data -= DeltaWM_W_manual
            #     model_WM.M.data -= DeltaWM_M_manual            
        # if torch.all(torch.isfinite(DeltawCw_W_manual)) and torch.all(torch.isfinite(DeltawCw_w_manual)):
        #     #print(43)
        #     #print(f'first %s' % cost_wCw.detach())
        #     model_wCw.W.data += DeltawCw_W_manual
        #     model_wCw.w.data += DeltawCw_w_manual
        #     Y_wCw = model_wCw(x_curr)
        #     cost_wCw = similarity_matching_cost(x_curr, Y_wCw)
        #     #print(f'second %s' % cost_wCw.detach())
        #     #print(model_wCw.w.data)
        #     #print(DeltawCw_M_manual)
        #     #print(DeltawCw_w_manual)
        #     if torch.any(torch.isnan(cost_wCw)):
        #         print(44)
        #         model_wCw.W.data -= DeltawCw_W_manual
        #         model_wCw.w.data -= DeltawCw_w_manual
        #         Y_wCw = model_wCw(x_curr)
        #         cost_wCw0 = similarity_matching_cost(x_curr, Y_wCw)
        #         #print(cost_wCw.detach(),' , ',cost_wCw0.detach())
        #         #print(model_wCw.w.data)

    if auto == 1:
        model_WM.W.data += torch.sqrt(torch.tensor(dt)) * xis
        model_WM.M.data += torch.sqrt(torch.tensor(dt)) * zetas
        # model_wCw.W.data += torch.sqrt(torch.tensor(dt)) * xis
        # model_wCw.w.data += torch.tensor(dt) * noise_w

    if epoch % step == 0 and epoch>1000:
        y=model_WM(X[sel_inx,:])
        yx =y.detach()
        Yt_WM[:,nn,:] = yx.t()
        # y=model_wCw(X[sel_inx,:])
        # yx =y.detach()
        # Yt_wCw[:,nn,:] = yx.t()
        nn += 1

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Cost: {cost_WM.item()}')
        # print(f'Epoch {epoch}, Cost: {cost_wCw.item()}')

##############################################################################
##############################################################################

selYInx = 100  # np.random.choice(range(num_sel), 1, replace=False)
y_WM_np = Yt_WM[:,:-200,selYInx]
# Assuming y is the output from the last epoch
for i in range(output_dim):
    plt.plot(y_WM_np[i , :], label=f'Output dimension {i+1}', alpha=0.6)
plt.show()
# y_wCw_np = Yt_wCw[:,:-200,selYInx]
# # Assuming y is the output from the last epoch
# for i in range(output_dim):
#     plt.plot(y_wCw_np[i , :], label=f'Output dimension {i+1}', alpha=0.6)
# plt.show()

y_wCw_np1 = Yt_WM[:,100,:]
# Assuming y is the output from the last epoch
plt.scatter(y_wCw_np1[0, :], y_wCw_np1[1, :], label=f'Output dimension {i+1}', alpha=0.6, color='blue')
plt.scatter(y_wCw_np1[1, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='red')
plt.scatter(y_wCw_np1[0, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='green')
plt.show()
y_wCw_np1 = Yt_WM[:,180,:]
# Assuming y is the output from the last epoch
plt.scatter(y_wCw_np1[0, :], y_wCw_np1[1, :], label=f'Output dimension {i+1}', alpha=0.6, color='blue')
plt.scatter(y_wCw_np1[1, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='red')
plt.scatter(y_wCw_np1[0, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='green')
plt.show()

##############################################################################
##############################################################################

# smSelect = np.random.choice(range(num_sel), 20, replace=False)

# # Selecting outputs at two time points
# time1 = 100  # Adjusted for Python indexing
# time2 = 2100#round(tot_iter/step)-1  # Assuming tot_iter > 2000
# Y1sel = Yt_WM[:-1, time1 , smSelect]
# Y2sel = Yt_WM[:-1, time2 , smSelect]
# # Perform hierarchical clustering to order the indices
# D = pdist(Y1sel.T, 'euclidean')
# tree = linkage(D, 'average')
# # Corrected function name for optimal leaf ordering
# leafOrder = leaves_list(optimal_leaf_ordering(tree, D))
# # Plotting the similarity matrices
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# cmap = 'RdBu'  # Assuming RdBuMap is a colormap name in MATLAB; use equivalent in matplotlib
# # Similarity matrix at time1
# sim_matrix_1 = np.dot(Y1sel[:, leafOrder].T, Y1sel[:, leafOrder])
# im = axes[0].imshow(sim_matrix_1, cmap=cmap, vmin=-6, vmax=6)
# axes[0].set_xticks([0, 9, 19])
# axes[0].set_yticks([0, 9, 19])
# axes[0].set_xlabel('Stimuli')
# axes[0].set_ylabel('Stimuli')
# axes[0].set_title('Time 1 Similarity Matrix')
# # Similarity matrix at time2
# sim_matrix_2 = np.dot(Y2sel[:, leafOrder].T, Y2sel[:, leafOrder])
# im = axes[1].imshow(sim_matrix_2, cmap=cmap, vmin=-6, vmax=6)
# axes[1].set_xticks([0, 9, 19])
# axes[1].set_yticks([])
# axes[1].set_xlabel('Stimuli')
# axes[1].set_title('Time 2 Similarity Matrix')
# # Colorbar
# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
# cbar.set_label('Similarity')
# plt.tight_layout()
# plt.show()

# # Selecting outputs at two time points
# time1 = 1  # Adjusted for Python indexing
# time2 = 2000#round(tot_iter/step)-1  # Assuming tot_iter > 2000
# Y1sel = Yt_wCw[:, time1 , smSelect]
# Y2sel = Yt_wCw[:, time2 , smSelect]
# # Perform hierarchical clustering to order the indices
# D = pdist(Y1sel.T, 'euclidean')
# tree = linkage(D, 'average')
# # Corrected function name for optimal leaf ordering
# leafOrder = leaves_list(optimal_leaf_ordering(tree, D))
# # Plotting the similarity matrices
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# cmap = 'RdBu'  # Assuming RdBuMap is a colormap name in MATLAB; use equivalent in matplotlib
# # Similarity matrix at time1
# sim_matrix_1 = np.dot(Y1sel[:, leafOrder].T, Y1sel[:, leafOrder])
# im = axes[0].imshow(sim_matrix_1, cmap=cmap, vmin=-6, vmax=6)
# axes[0].set_xticks([0, 9, 19])
# axes[0].set_yticks([0, 9, 19])
# axes[0].set_xlabel('Stimuli')
# axes[0].set_ylabel('Stimuli')
# axes[0].set_title('Time 1 Similarity Matrix')
# # Similarity matrix at time2
# sim_matrix_2 = np.dot(Y2sel[:, leafOrder].T, Y2sel[:, leafOrder])
# im = axes[1].imshow(sim_matrix_2, cmap=cmap, vmin=-6, vmax=6)
# axes[1].set_xticks([0, 9, 19])
# axes[1].set_yticks([])
# axes[1].set_xlabel('Stimuli')
# axes[1].set_title('Time 2 Similarity Matrix')
# # Colorbar
# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
# cbar.set_label('Similarity')
# plt.tight_layout()
# plt.show()


##############################################################################
##############################################################################

def compute_diffusion_constants(Y, plot=False):
    """
    Computes the diffusion constant for each component of Y across time.
    
    Parameters:
    - Y: A torch.Tensor or np.ndarray with shape [time_points, components]
         representing the output of a network across time for different components.
    - plot: A boolean. If True, plots the MSD and linear fits.
    
    Returns:
    - An np.ndarray of diffusion constants for each component.
    """
    # Ensure Y is a numpy array if it's a PyTorch tensor
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().numpy()
    
    components, time_points = Y.shape  # Extract dimensions

    # Define the linear fit function for MSD = 2Dt
    def linear_fit(t, D):
        return 2 * D * t

    # Prepare to store the diffusion constants for each component
    Ds = np.zeros(components)

    # Iterate over each component to calculate its diffusion constant
    for component in range(components):
        # Extract the series for the current component
        Y_component = Y[component,:] 
        
        # Calculate the mean squared displacement (MSD) for each time point relative to the first time point
        MSD = np.square(Y_component - Y_component[0])
        # Time steps (assuming each step is 1 unit)
        time_steps = np.arange(time_points)

        # Fit the MSD data to the linear model to find the diffusion constant
        params, _ = curve_fit(linear_fit, time_steps, MSD)
        
        # Store the diffusion constant for this component
        Ds[component] = params[0] #np.mean(MSD) #

    if plot:
        # Plot the MSD and fitted lines for visualization
        plt.figure(figsize=(10, 6))
        for component in range(components):
            plt.plot(time_steps, linear_fit(time_steps, Ds[component]), label=f'Component {component+1}, D={Ds[component]:.2f}')
        plt.xlabel('Time')
        plt.ylabel('MSD')
        plt.legend()
        plt.title('Mean Squared Displacement and Fitted Diffusion Constants for Each Component')
        plt.show()
    
    return Ds

# Example usage:
# Assuming Y is a tensor with shape [time_points, components]
# Y = torch.randn(100, 3)  # Replace with your actual tensor
Ds = compute_diffusion_constants(y_WM_np, plot=False)
print("Diffusion constants for each component:", Ds)
# Ds = compute_diffusion_constants(y_wCw_np, plot=False)
# print("Diffusion constants for each component:", Ds)


# n, time, trials = Yt_WM.shape  # Example dimensions
# correlation_matrices = np.zeros((time, n, n))
# d12 = np.zeros((time, n))
# d13 = np.zeros((time, n))
# d23 = np.zeros((time, n))

# for t in range(time):
#     data_at_t = Yt_wCw[:, t, :]
#     correlation_matrices[t] = np.corrcoef(data_at_t)
#     d12[t] = correlation_matrices[t][0,1]
#     d13[t] = correlation_matrices[t][0,2]
#     d23[t] = correlation_matrices[t][1,2]

# # Calculate the average correlation matrix across all time points
# average_correlation_matrix = np.nanmean(correlation_matrices, axis=0)

# # Plot the average correlation matrix
# plt.imshow(average_correlation_matrix, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Average Correlation Matrix Across Time')
# plt.xlabel('n')
# plt.ylabel('n')
# plt.show()

# # Plot the average correlation matrix
# plt.imshow(average_correlation_matrix-C_target_np, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Average Correlation Matrix Across Time')
# plt.xlabel('n')
# plt.ylabel('n')
# plt.show()



n, time, trials = Yt_WM.shape  # Example dimensions
correlation_matrices = np.zeros((time, n, n))
c12 = np.zeros((time, n))
c13 = np.zeros((time, n))
c23 = np.zeros((time, n))

for t in range(time):
    data_at_t = Yt_WM[:, t, :]
    correlation_matrices[t] = np.corrcoef(data_at_t)
    c12[t] = correlation_matrices[t][0,1]
    c13[t] = correlation_matrices[t][0,2]
    c23[t] = correlation_matrices[t][1,2]

# Calculate the average correlation matrix across all time points
average_correlation_matrix = np.nanmean(correlation_matrices, axis=0)

# Plot the average correlation matrix
plt.imshow(average_correlation_matrix, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Average Correlation Matrix Across Time')
plt.xlabel('n')
plt.ylabel('n')
plt.show()


plt.plot(c12[:1000])
plt.plot(c13[:1000])
plt.plot(c23[:1000])
plt.title('WM model')
plt.show()

# plt.plot(d12[:1000])
# plt.plot(d13[:1000])
# plt.plot(d23[:1000])
# plt.title('wCw model')
# plt.show()


M = model_WM.M.detach().numpy()
w = 1 / np.sqrt(np.diag(M))
w = np.diag(w)
plt.imshow(w @ M @ w, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
plt.colorbar()


volume = estimate_volume_convex_hull(y_WM_np.T)
print(f"Estimated Volume: {volume}")
