import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
import numpy as np
from scipy.optimize import curve_fit
import sys
from scipy.spatial import ConvexHull, qhull



##############################################################################
##############################################################################

def estimate_volume_convex_hull(points):

    try:
        hull = ConvexHull(points)
        return hull.volume
    except scipy.spatial.qhull.QhullError:
        print("Error computing the convex hull. The points may be collinear or not span the entire space.")
        return None

def compute_diffusion_constants(Y, plot=False):
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().numpy()
    
    components, time_points = Y.shape  # Extract dimensions

    def linear_fit(t, D):
        return 2 * D * t
    Ds = np.zeros(components)
    for component in range(components):
        Y_component = Y[component,:]         
        MSD = np.square(Y_component - Y_component[0])
        time_steps = np.arange(time_points)
        params, _ = curve_fit(linear_fit, time_steps, MSD)
        
        Ds[component] = params[0] #np.mean(MSD) #

    Dsm=np.mean(Ds)
    
    return Dsm

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

def similarity_matching_cost_0(x, y):
    T = x.shape[0]  # Number of samples
    S_x = torch.matmul(x, x.t())
    S_y = torch.matmul(y, y.t())
    cost = torch.mean((S_x - S_y) ** 2) / (T ** 2)
    return cost

def similarity_matching_cost(x, model, C):
    T = x.shape[0]  # Number of samples
    Y = model(x)
    cost = similarity_matching_cost_0(x, Y)

    M = model.M
    M_diag = torch.diag(M)
    M_ii = M_diag.unsqueeze(1)  # Make it a column vector
    M_jj = M_diag.unsqueeze(0)  # Make it a row vector
    E = torch.zeros_like(M)
    mask = torch.eye(M.size(0), dtype=torch.bool)
    E[~mask] = (M[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask]) - (C[~mask])

    cost += torch.norm(E, 'fro')

    return cost

############################################################################## 
##############################################################################


def Simulate_Drift(X, stdW , stdM, rho, auto, model_WM, input_dim = 10,output_dim = 3):

    #input_dim = 10  # Example dimensions
    #output_dim = 3
    tot_iter = 50000
    #syn_noise_std = 0.2
    learnRate = 0.1
    dt = learnRate
    #stdW = syn_noise_std
    #stdM = syn_noise_std
    num_sel = 200 # randomly selected samples used to calculate the drift and diffusion constants
    step = 10    # store every 10 updates
    time_points = round(tot_iter / step)
    sel_inx = np.random.permutation(num_samples)[:num_sel]

    Yt_WM = np.zeros((output_dim, time_points, num_sel))
    Ds_v = np.zeros((time_points-200))
    volume_v = np.zeros((time_points-200))

    #rho = 0.3 
    C_target_np = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])  # Target correlation matrix
    C_target = torch.tensor(C_target_np, dtype=torch.float32)
    C_target[~torch.eye(3, dtype=bool)] *= -1  # Only modify non-diagonal elements

    #model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim)
    optimizer_WM = torch.optim.SGD(model_WM.parameters(), lr=learnRate)
    DeltaWM_W_manual = torch.nn.Parameter(torch.randn(output_dim, input_dim))
    DeltaWM_M_manual = torch.nn.Parameter(torch.eye(output_dim))
    nn = 0

    for epoch in range(tot_iter):  # Number of epochs

        # Randomly select one sample
        curr_inx = torch.randint(0, num_samples, (100,))
        x_curr = X[curr_inx,:]  # Current input sample
        Y_WM = model_WM(x_curr)

        optimizer_WM.zero_grad()
        cost_WM = similarity_matching_cost(x_curr, model_WM, C_target)
        cost_WM.backward()

        if auto == 1:
            autoWM_grad_W = model_WM.W.grad.clone()
            autoWM_grad_M = model_WM.M.grad.clone()
            optimizer_WM.step()

        # Generate noise matrices
        xis = torch.randn(output_dim, input_dim) * stdW 
        zetas = torch.randn(output_dim, output_dim) * stdM 

        # Update W and M with noise
        DeltaWM_W_manual = dt * (torch.matmul(Y_WM.t(), x_curr) / x_curr.size(0) - model_WM.W)+ torch.sqrt(torch.tensor(dt)) * xis  # y_i x_j - W_ij

        M = model_WM.M
        C = C_target
        M_diag = torch.diag(M)
        M_ii = M_diag.unsqueeze(1)  # Make it a column vector
        M_jj = M_diag.unsqueeze(0)  # Make it a row vector
        E = torch.zeros_like(M)
        mask = torch.eye(M.size(0), dtype=torch.bool)
        E[~mask] = (M[~mask] / (M_ii * M_jj)[~mask]) - (C[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask])
        c_alpha = 0

        DeltaWM_M_manual = dt * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model_WM.M- c_alpha * E)+ torch.sqrt(torch.tensor(dt)) * zetas  # y_i y_j - M_ij torch.sqrt(torch.tensor(dt))

        if auto == 0:
            if torch.all(torch.isfinite(DeltaWM_W_manual)) and torch.all(torch.isfinite(DeltaWM_M_manual)):
                model_WM.W.data += DeltaWM_W_manual
                model_WM.M.data += DeltaWM_M_manual

        if auto == 1:
            model_WM.W.data += torch.sqrt(torch.tensor(dt)) * xis
            model_WM.M.data += torch.sqrt(torch.tensor(dt)) * zetas

        if epoch % step == 0 and epoch>1000:
            y=model_WM(X[sel_inx,:])
            yx =y.detach()
            Yt_WM[:,nn,:] = yx.t()
            nn += 1

        if epoch % 10000 == 0:
            print(f'Epoch {epoch}, Cost: {cost_WM.item()}')

    for inn in range(Yt_WM.shape[2]):
        selYInx = inn#100  # np.random.choice(range(num_sel), 1, replace=False)
        y_WM_np = Yt_WM[:,:-200,selYInx]
        Ds_v[inn] = compute_diffusion_constants(y_WM_np, plot=False)
        volume_v[inn] = estimate_volume_convex_hull(y_WM_np.T)

    Ds = np.mean(Ds_v,axis=0)
    volume = np.mean(volume_v,axis=0)

    return Ds, volume, Yt_WM, model_WM

##############################################################################
##############################################################################
##############################################################################
#############################  MAIN PART  ####################################
##############################################################################
input_dim=10
output_dim=3
num_samples = 5000
auto = 0

eigenvalues = [4.5, 3.5, 1]  + [0.01] * (input_dim - 3)  # Eigenvalues for the covariance matrix
X = generate_PSP_input_torch(eigenvalues, input_dim, num_samples)

model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim)
#Ds0, volume0, y_WM_np0, model_WM0 =  Simulate_Drift(X, 0, 0, 0, auto, model_WM)
#avg_Ds0 = np.mean(Ds0)
#print(f"stdW: {0:.2f}, stdM: {0:.2f}, rho: {0:.2f}, Avg Ds: {np.mean(avg_Ds0):.4f}, Volume: {np.mean(volume0):.4f}")

stdWs = np.linspace(0, 0.2, 2)
stdMs = np.linspace(0, 0.2, 2)
rhos = np.linspace(-0.5, 0.5, 3)
# Prepare to store the results
#Ds_results = np.zeros((len(stdWs), len(stdMs), len(rhos)))
#volume_results = np.zeros_like(Ds_results)

for i, stdW in enumerate(stdWs):
    for j, stdM in enumerate(stdMs):
        for k, rho in enumerate(rhos):
            if k == 0 and i == 0 and j == 0:
                Ds0, volume0, Yt_WM0, model_WM0 =  Simulate_Drift(X, 0, 0, rho, auto, model_WM)

            Ds, volume, Yt_WM, model_WM =  Simulate_Drift(X, stdW, stdM, rho, auto, model_WM0)
            #avg_Ds = np.mean(Ds)
            #Ds_results[i, j, k] = avg_Ds
            #volume_results[i, j, k] = volume
            #print(f"stdW: {stdW:.2f}, stdM: {stdM:.2f}, rho: {rho:.2f}, Avg Ds: {avg_Ds:.4f}, Volume: {volume:.4f}")

if 1==1:

    sel_inx=100
    #y=model_WM0(X[sel_inx,:])
    #yx =y.detach()
    #Yt_WM = yx.t()
    #selYInx = 100  # np.random.choice(range(num_sel), 1, replace=False)
    y_WM_np = Yt_WM[:,:-200,sel_inx]
    # Assuming y is the output from the last epoch
    for i in range(output_dim):
        plt.plot(y_WM_np[i , :], label=f'Output dimension {i+1}', alpha=0.6)
    plt.show()
    # y_wCw_np = Yt_wCw[:,:-200,selYInx]
    # # Assuming y is the output from the last epoch
    # for i in range(output_dim):
    #     plt.plot(y_wCw_np[i , :], label=f'Output dimension {i+1}', alpha=0.6)
    # plt.show()

    y_wCw_np1 = Yt_WM[100,:].T
    # Assuming y is the output from the last epoch
    plt.scatter(y_wCw_np1[0, :], y_wCw_np1[1, :], label=f'Output dimension {i+1}', alpha=0.6, color='blue')
    plt.scatter(y_wCw_np1[1, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='red')
    plt.scatter(y_wCw_np1[0, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='green')
    plt.show()
    y_wCw_np1 = Yt_WM[150,:].T
    # Assuming y is the output from the last epoch
    plt.scatter(y_wCw_np1[0, :], y_wCw_np1[1, :], label=f'Output dimension {i+1}', alpha=0.6, color='blue')
    plt.scatter(y_wCw_np1[1, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='red')
    plt.scatter(y_wCw_np1[0, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='green')
    plt.show()

    ##############################################################################
    ##############################################################################

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
