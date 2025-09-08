import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.stats import entropy

# Ensure all tensors are created on the CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

##############################################################################
# Helper function to generate input data using PyTorch on the specified device
def generate_PSP_input_torch(input_cov_eigens, input_dim, num_samples, device):
    Q, _ = torch.linalg.qr(torch.randn(input_dim, input_dim, device=device))
    C = Q @ torch.diag(torch.tensor(input_cov_eigens, device=device)) @ Q.T
    mean = torch.zeros(input_dim, device=device)
    X = torch.distributions.MultivariateNormal(mean, C).sample((num_samples,))
    return X

def create_block_correlation_matrix(output_dim, rho):
    block_size = output_dim // 2  # Size of each block
    block_matrix = torch.zeros((output_dim, output_dim), device=device)

    block_matrix[:block_size, :block_size].fill_(rho)
    block_matrix[block_size:, block_size:].fill_(rho)
    torch.diagonal(block_matrix).fill_(1)

    return block_matrix

def matrix_sqrt(C):
    C = 0.5 * (C + C.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    C_sqrt = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    D_sqrt = torch.diag(C_sqrt @ C_sqrt).sqrt()
    C_sqrt = C_sqrt / D_sqrt.unsqueeze(0)
    return C_sqrt

##############################################################################
##############################################################################

def estimate_volume_convex_hull(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except qhull.QhullError:
        print("Error computing the convex hull. The points may be collinear or not span the entire space.")
        return None

def compute_diffusion_constants(Y, plot=False):
    Y = Y.detach()
    components, time_points = Y.shape
    Ds = torch.zeros(components, device=Y.device)

    for component in range(components):
        Y_component = Y[component, :]
        MSD = torch.pow(Y_component - Y_component[0], 2)
        time_steps = torch.arange(time_points, device=Y.device).float()
        xmean = time_steps.mean()
        ymean = MSD.mean()
        A = (torch.sum(time_steps * MSD) - time_points * xmean * ymean) / (torch.sum(time_steps ** 2) - time_points * xmean ** 2)
        D = A / 2
        Ds[component] = D

    Dsm = Ds.mean().item()
    return Ds

def compute_entropy_from_histogram(distances, bins='auto'):
    y = np.zeros(distances.shape[0])

    for i in range(distances.shape[0]):
        hist, _ = np.histogram(distances[i, :].cpu().numpy(), bins=bins, density=True)
        prob_dist = hist / np.sum(hist)
        y[i] = entropy(prob_dist)

    return np.mean(y)

##############################################################################
# Neural Network Model Definition
class SimilarityMatchingNetwork_WM(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(SimilarityMatchingNetwork_WM, self).__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim, device=device))
        self.M = nn.Parameter(torch.eye(output_dim, device=device))
        
    def forward(self, x):
        M_noisy = self.M
        W_noisy = self.W
        I = torch.eye(M_noisy.size(0), device=device)

        det_A = torch.det(M_noisy)
        is_singular = torch.isclose(det_A, torch.tensor(0.0, device=device))

        if not is_singular:
            M_inv = torch.inverse(M_noisy)
        else:
            M_inv = torch.full(M_noisy.shape, float('nan'), device=device)

        y = M_inv @ W_noisy @ x.t()
        return y.t()

##############################################################################
##############################################################################

def similarity_matching_cost_0(x, y):
    T = x.shape[0]
    S_x = torch.matmul(x, x.t())
    S_y = torch.matmul(y, y.t())
    cost = torch.mean((S_x - S_y) ** 2) / (T ** 2)
    return cost

def similarity_matching_cost(x, model, C):
    T = x.shape[0]
    Y = model(x)
    cost = similarity_matching_cost_0(x, Y)

    M = model.M
    M_diag = torch.diag(M)
    M_ii = M_diag.unsqueeze(1)
    M_jj = M_diag.unsqueeze(0)
    E = torch.zeros_like(M)
    mask = torch.eye(M.size(0), dtype=torch.bool)
    E[~mask] = (M[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask]) - (C[~mask])

    cost += torch.norm(E, 'fro')
    return cost

##############################################################################
##############################################################################

def Simulate_Drift(X, stdW, stdM, rho, auto, model_WM, input_dim, output_dim):
    X = X.to(device)
    model_WM.to(device)
    tot_iter = 100000
    learnRate = 0.05
    dt = learnRate
    num_sel = 200
    step = 10
    num_samples = X.shape[0]
    time_points = round(tot_iter / step)
    sel_inx = torch.randperm(num_samples)[:200].to(device)
    Yt_WM = torch.zeros(output_dim, time_points, num_sel, device=device)
    Ds_v = torch.zeros(num_sel, output_dim, device=device)
    volume_v = torch.zeros(num_sel, device=device)

    C_target = torch.full((output_dim, output_dim), rho, device=device)
    noise = torch.randn(output_dim, output_dim, device=device) * 0.01
    torch.diagonal(noise).fill_(0)
    C_target += noise
    torch.diagonal(C_target).fill_(1)
    C_target = torch.where(torch.eye(output_dim, device=device, dtype=torch.bool), C_target, -C_target)
    
    DeltaWM_W_manual = torch.nn.Parameter(torch.randn(output_dim, input_dim, device=device))
    DeltaWM_M_manual = torch.nn.Parameter(torch.eye(output_dim, device=device))
    nn = 0

    for epoch in range(tot_iter):
        curr_inx = torch.randint(0, num_samples, (1000,), device=device)
        x_curr = X[curr_inx, :]
        Y_WM = model_WM(x_curr)

        if auto == 1:
            autoWM_grad_W = model_WM.W.grad.clone()
            autoWM_grad_M = model_WM.M.grad.clone()
            optimizer_WM.step()

        xis = torch.randn(output_dim, input_dim, device=device) * stdW
        zetas = torch.randn(output_dim, output_dim, device=device) * stdM

        DeltaWM_W_manual = dt * (torch.matmul(Y_WM.t(), x_curr) / x_curr.size(0) - model_WM.W) + torch.sqrt(torch.tensor(dt, device=device)) * xis

        M = model_WM.M
        C = C_target
        M_diag = torch.diag(M)
        M_ii = M_diag.unsqueeze(1)
        M_jj = M_diag.unsqueeze(0)
        E = torch.zeros_like(M, device=device)
        mask = torch.eye(M.size(0), dtype=torch.bool, device=device)
        E[~mask] = (M[~mask] / (M_ii * M_jj)[~mask]) - (C[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask])
        c_alpha = 3

        DeltaWM_M_manual = dt * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model_WM.M - c_alpha * E) + torch.sqrt(torch.tensor(dt, device=device)) * zetas

        if auto == 0:
            if torch.all(torch.isfinite(DeltaWM_W_manual)) and torch.all(torch.isfinite(DeltaWM_M_manual)):
                model_WM.W.data += DeltaWM_W_manual
                model_WM.M.data += DeltaWM_M_manual

        if auto == 1:
            model_WM.W.data += torch.sqrt(torch.tensor(dt, device=device)) * xis
            model_WM.M.data += torch.sqrt(torch.tensor(dt, device=device)) * zetas

        if epoch % step == 0:
            y = model_WM(X[sel_inx, :])
            yx = y.detach()
            Yt_WM[:, nn, :] = yx.t()
            nn += 1

        if epoch % 10000 == 0:
            print(f'Epoch {epoch}, Cost: {0}')

    for inn in range(Yt_WM.shape[2]):
        selYInx = inn
        y_WM_np = Yt_WM[:, :-200, selYInx]
        Ds_v[inn, :] = compute_diffusion_constants(y_WM_np, plot=False)
        volume_v[inn] = compute_entropy_from_histogram(y_WM_np)

    Ds = torch.mean(Ds_v, axis=0)
    volume = torch.mean(volume_v, axis=0)
    
    return Ds_v, volume, Yt_WM, model_WM

##############################################################################
##############################################################################
#############################  MAIN PART  ####################################
##############################################################################
##############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
input_dim = 1
output_dim = 2
num_samples = 1000
auto = 0

X = torch.randn(num_samples, input_dim, device=device)
for ix in range(input_dim):
    current_mean = X[:, ix].mean()
    X[:, ix] += (ix + 1) - current_mean

model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim, device)

stdWs = torch.linspace(0, 0.05, 2, device=device)
stdMs = torch.linspace(0, 0.05, 2, device=device)
rhos = torch.linspace(-0.8, 0.8, 21, device=device)

Ds_results = torch.zeros((len(stdWs), len(stdMs), len(rhos)), device=device)
volume_results = torch.zeros_like(Ds_results)
#Similarity_results = torch.zeros((len(stdWs), len(stdMs), Similarity0.shape[0], Similarity0.shape[1], Similarity0.shape[2]), device=device)

# for i, stdW in enumerate(stdWs):
#     for j, stdM in enumerate(stdMs):
i=0
stdW=0.05
j=0
stdM=0.0
for k, rho in enumerate(rhos):
    if k == 0 and i == 0 and j == 0:
        Ds0, volume0, Yt_WM0, model_WM0 =  Simulate_Drift(X, 0, 0, rho, auto, model_WM, input_dim,output_dim)
        Yt_WM_storage = np.zeros((len(stdWs), len(stdMs), len(rhos), Yt_WM0.shape[0], Yt_WM0.shape[1], Yt_WM0.shape[2]))
    if i == 0 and j == 0:
        Ds0, volume0, Yt_WM0, model_WM0 =  Simulate_Drift(X, 0, 0, rho, auto, model_WM, input_dim,output_dim)

    Ds, volume, Yt_WM, model_WM =  Simulate_Drift(X, stdW, stdM, rho, auto, model_WM0, input_dim,output_dim)
    avg_Ds = torch.mean(Ds)
    Ds_results[i, j, k] = avg_Ds
    volume_results[i, j, k] = volume
    print(f"stdW: {stdW:.2f}, stdM: {stdM:.2f}, rho: {rho:.2f}, Avg Ds: {avg_Ds:.4f}, Volume: {volume:.4f}")
    Yt_WM_storage[i, j, k] = Yt_WM

if 1==1:

    sel_inx=100
    #y=model_WM0(X[sel_inx,:])
    #yx =y.detach()
    #Yt_WM = yx.t()
    #selYInx = 100  # np.random.choice(range(num_sel), 1, replace=False)
    y_WM_np = Yt_WM[:,:-200,sel_inx]
    # Assuming y is the output from the last epoch
    for i in range(output_dim):
        plt.plot(y_WM_np[i , ::40], label=f'Output dimension {i+1}', alpha=0.6)
    plt.show()
    # y_wCw_np = Yt_wCw[:,:-200,selYInx]
    # # Assuming y is the output from the last epoch
    # for i in range(output_dim):
    #     plt.plot(y_wCw_np[i , :], label=f'Output dimension {i+1}', alpha=0.6)
    # plt.show()

    y_wCw_np1 = Yt_WM[:,10,:].T
    # Assuming y is the output from the last epoch
    plt.scatter(y_wCw_np1[0, :], y_wCw_np1[1, :], label=f'Output dimension {i+1}', alpha=0.6, color='blue')
    plt.scatter(y_wCw_np1[1, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='red')
    plt.scatter(y_wCw_np1[0, :], y_wCw_np1[2, :], label=f'Output dimension {i+1}', alpha=0.6, color='green')
    plt.show()
    y_wCw_np1 = Yt_WM[:,15,:].T
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

    XX=X[1,:]
    noise = torch.randn(X.shape[0], X.shape[1], device=device) * 0.01  # Adjust 0.1 to control the noise level
    X_noisy = XX + noise
    YY = model_WM(X_noisy)
    n, trials = YY.shape  # Example dimensions
    correlation_May = torch.corrcoef(YY.T)
    # Plot the average correlation matrix
    plt.imshow(correlation_May.detach().numpy(), cmap='bwr', interpolation='none', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Average Correlation Matrix Across Time')
    plt.xlabel('n')
    plt.ylabel('n')
    plt.show()



M = model_WM.M
w = torch.inverse(torch.sqrt(torch.diag(torch.diag(M))))
#w = torch.diag(w)
#w = np.diag(w)
Cm=torch.matmul(torch.matmul(w.T , M) , w)#-C_target
plt.imshow(Cm.detach().cpu().numpy(), cmap='bwr', interpolation='none', vmin=-1, vmax=1)
plt.colorbar()


plt.plot(Ds_results[0,0,:])


D1 = Ds_results[0,0,:].unsqueeze(1)
D2 = D1.T

plt.imshow(torch.abs(D1 -D2))



data_a = torch.mean(Ds.T,axis=1)

D1 = data_a.unsqueeze(1)
D2 = D1.T
plt.imshow(torch.abs(D1 -D2))
plt.colorbar()


correlation_m = torch.corrcoef(data_at_t)
plt.imshow(correlation_m, cmap='bwr', interpolation='none')
plt.colorbar()


X_flat = Cm.flatten()
Y_flat = torch.abs(D1 -D2).flatten()

# Plotting
plt.scatter(X_flat.detach(), Y_flat.detach())
