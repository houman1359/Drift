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
from scipy.stats import entropy
from IPython.display import display, clear_output
import time


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

def compute_entropy_from_histogram(distances, bins='auto'):

    y = np.zeros(distances.shape[0])

    for i in range(distances.shape[0]):  
        hist, _ = np.histogram(distances[i,:], bins=bins, density=True)
        prob_dist = hist / np.sum(hist)
        y[i]=entropy(prob_dist)

    return np.mean(y)

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

###  1- dimensionality  2- conjunctiveness 3- feedforward corr vs recurrent corr.
    
class PlaceCellNetwork(nn.Module):
    def __init__(self, num_copies, input_dim, output_dim, MaxIter, dt, alpha = 0.0, lbd1 = 0.0, lbd2 = 0.0):
        super(PlaceCellNetwork, self).__init__()
        self.W = nn.Parameter(torch.randn(num_copies, output_dim, input_dim))
        self.M = nn.Parameter(torch.stack([torch.eye(output_dim) for _ in range(num_copies)]))
        self.b = nn.Parameter(torch.zeros(num_copies, output_dim))
        self.MaxIter = MaxIter
        self.dt = dt
        self.alpha = alpha
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        self.M_off_diag = self.M - torch.stack([torch.eye(output_dim) for _ in range(num_copies)])
        self.cumErr = float('inf')
        self.errTrack = torch.rand(num_copies, 5, 1, device=self.W.device)  # Adjust device to match the parameters

    def forward(self, X):
        print(self.M)
        print(self.M_off_diag)
        
        device = X.device
        batch_size = X.size(0)  
        num_copies = self.W.size(0)
        Y = torch.zeros(num_copies, batch_size, self.W.size(1), device=device)
        Yold = Y.clone()
        X_reshaped = X.unsqueeze(1)
        diag_M = torch.diagonal(self.M, dim1=-2, dim2=-1)  # This extracts the diagonal across all copies
        diag_M = diag_M.unsqueeze(-2)  # Now diag_M is [1, output_dim, 1]
  
        W_transposed = self.W.transpose(1, 2)  
        X_expanded = X.unsqueeze(0).expand(num_copies, -1, -1)  
        Wx = torch.bmm(X_expanded,W_transposed)

        for count in range(self.MaxIter):
            M_Y = torch.bmm(Yold, self.M_off_diag.transpose(1, 2))
            dt = max(self.dt / (1 + count / 10), 1e-2)  # Decaying learning rate
            du = -Yold + Wx - np.sqrt(self.alpha)  * self.b.unsqueeze(1) - M_Y
            uy = Yold + dt * du
            Y = torch.maximum((uy - self.lbd1) / (self.lbd2 + diag_M), torch.zeros_like(uy))
            err = torch.norm(Y - Yold, p='fro', dim=[1, 2]) / (torch.norm(Yold, p='fro', dim=[1, 2]) + 1e-10) / dt
            err_unsqueezed = err.unsqueeze(1).unsqueeze(2)  # Adjust shape for concatenation
            self.errTrack = torch.cat((err_unsqueezed, self.errTrack[:, :4]), dim=1)
            cumErr = torch.abs(torch.sum(torch.diff(self.errTrack, dim=1), dim=1))

            if torch.all(err < 1e-4) and torch.all(cumErr < 1e-8):
                break

            Yold = Y.clone()

        return Y.squeeze(1)

##############################################################################
##############################################################################

def similarity_matching_cost_0(x, y, alpha = 0.0, beta_1 = 0.0, beta_2 = 0.0):
    T = x.shape[0]  # Number of samples
    S_x = torch.matmul(x, x.t())
    S_y = torch.matmul(y, y.t())
    cost = torch.mean((S_x - S_y) ** 2) / (T ** 2)
    E = torch.ones((T, T), device=x.device)  # Ensure E is on the same device as x and y

    alpha_E = alpha**2 * E
    cost = torch.mean((S_x - S_y - alpha_E) ** 2) / (T ** 2)

    l1_reg = torch.sum(torch.abs(y), dim=1).mean() / T
    l2_reg = torch.sum(y ** 2, dim=1).mean() / T
    total_cost = cost + 2 * beta_1 * l1_reg + beta_2 * l2_reg
    
    return cost

def similarity_matching_cost(x, model, C, alpha=0.0, beta_1=0.0, beta_2=0.0):
    T = x.shape[0]  # Number of samples
    Y = model(x)
    cost = similarity_matching_cost_0(x, Y, alpha, beta_1, beta_2)

    M = model.M
    M_diag = torch.diag(M)
    M_ii = M_diag.unsqueeze(1)  # Make it a column vector
    M_jj = M_diag.unsqueeze(0)  # Make it a row vector
    E = torch.zeros_like(M)
    mask = torch.eye(M.size(0), dtype=torch.bool)
    E[~mask] = (M[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask]) - (C[~mask])

    cost += alpha * torch.norm(E, 'fro')

    return cost

##############################################################################
##############################################################################

def Simulate_Drift_NL(X, stdW , stdM, rho, batch_size, model, input_dim, output_dim, lr, alpha=0.0, beta_1=0.0, beta_2=0.0):

    num_sel = batch_size # randomly selected samples used to calculate the drift and diffusion constants
    step = 10#10    # store every 10 updates
    time_points = round(tot_iter / step)
    sel_inx = np.random.permutation(num_samples)[:num_sel]

    Yt_WM = np.zeros((output_dim, time_points, num_sel))
    Ds_v = np.zeros((time_points-0))
    volume_v = np.zeros((time_points-0))
    Similarity = np.zeros((time_points-0,output_dim,output_dim))

    #rho = 0.0
    #C_target_np = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])  # Target correlation matrix
    #C_target = torch.tensor(C_target_np, dtype=torch.float32)
    C_target_np = np.full((output_dim, output_dim), rho)
    noise = np.random.normal(0, 0.01, size=(output_dim, output_dim))
    noise[np.arange(output_dim), np.arange(output_dim)] = 0  # Zero out diagona l noise
    C_target_np += noise
    np.fill_diagonal(C_target_np, 1)
    C_target = torch.tensor(C_target_np, dtype=torch.float32)
    C_target[~torch.eye(output_dim, dtype=bool)] *= -1  # Only modify non-diagonal elements
    upper_tri_A = torch.triu(C_target)
    C_target = upper_tri_A + upper_tri_A.t() - torch.diag(torch.diag(upper_tri_A))

    #model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim)
    optimizer_WM = torch.optim.SGD(model.parameters(), lr=lr)
    DeltaWM_W_manual = torch.nn.Parameter(torch.randn(output_dim, input_dim))
    DeltaWM_M_manual = torch.nn.Parameter(torch.eye(output_dim))
    nn = 0
    
    #fig, ax = plt.subplots()    

    for epoch in range(tot_iter):  # Number of epochs

        start_time = time.time()
        # Randomly select one sample
        curr_inx = torch.randint(0, num_samples, (batch_size,)) #torch.tensor([1])
        x_curr = X[curr_inx,:] 
        Y_WM = model(x_curr)
        Y_WM = Y_WM.transpose(1, 2)
        x_curr = x_curr.unsqueeze(0).expand(num_copies, -1, -1)

        xis = torch.randn_like(model.W) * stdW
        zetas = torch.randn_like(model.M) * stdM * 0
        xi_b = torch.randn_like(model.b) * stdM * 0
        

        # M = model.M
        # C = C_target
        # M_diag = torch.diag(M)
        # M_ii = M_diag.unsqueeze(1) 
        # M_jj = M_diag.unsqueeze(0) 
        # E = torch.zeros_like(M)
        # mask = torch.eye(M.size(0), dtype=torch.bool)
        # E[~mask] = (M[~mask] / (M_ii * M_jj)[~mask]) - (C[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask])
        # c_alpha = 0.0
        # DeltaWM_M_manual = dt * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model.M- c_alpha * E)+ torch.sqrt(torch.tensor(dt)) * zetas  # y_i y_j - M_ij torch.sqrt(torch.tensor(dt))

        # DeltaW = lr * (torch.matmul(Y_WM.t(), x_curr) / x_curr.size(0) - model.W) + torch.sqrt(torch.tensor(lr)) * xis
        # DeltaM = lr * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model.M) + torch.sqrt(torch.tensor(lr)) * zetas
        # Deltab = lr * (np.sqrt(alpha) * torch.mean(Y_WM, dim=0) - model.b) + torch.sqrt(torch.tensor(lr)) * xi_b

        # Compute updates for W, M, and b with noise
        #DeltaW = lr * (torch.bmm(Y_WM.transpose(1, 2), x_curr.unsqueeze(1).expand(-1, num_copies, -1)) / x_curr.size(0) - model.W) + noise_W
        DeltaW = lr * (torch.bmm(Y_WM, x_curr) / batch_size - model.W) + torch.sqrt(torch.tensor(lr)) * xis
        DeltaM = lr * (torch.bmm(Y_WM, Y_WM.transpose(1, 2)) / batch_size - model.M) + zetas
        Deltab = lr * (torch.mean(Y_WM, dim=2) - model.b) + xi_b


        # DeltaW = torch.where(torch.isinf(DeltaW), torch.tensor(0.0), DeltaW)
        # DeltaM = torch.where(torch.isinf(DeltaM), torch.tensor(0.0), DeltaM)
        # Deltab = torch.where(torch.isinf(Deltab), torch.tensor(0.0), Deltab)
 
        # DeltaW = torch.where(torch.isnan(DeltaW), torch.tensor(0.0), DeltaW)
        # DeltaM = torch.where(torch.isnan(DeltaM), torch.tensor(0.0), DeltaM)
        # Deltab = torch.where(torch.isnan(Deltab), torch.tensor(0.0), Deltab)

        with torch.no_grad():
            if torch.all(torch.isfinite(DeltaW)) and torch.all(torch.isfinite(DeltaM)) and torch.all(torch.isfinite(Deltab)):
                model.W += DeltaW
                model.M += DeltaM
                model.b += Deltab
            #else:
                #print(DeltaM)
                #print(DeltaM)
                #print(Deltab)

        # if epoch % step == 0: #and epoch>1000:
        #     y=model(X)
        #     print(y.shape)
        #     yx =y.detach()
        #     Yt_WM[:,nn,:] = yx.t()
        #     nn += 1

        # cost = similarity_matching_cost(x_curr, model, C_target, alpha, beta_1, beta_2)
        # ax.plot(epoch, cost.detach().numpy(), 'b.')  # 'b.' plots a blue dot
        # ax.set_title('Cost over Epochs')
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Cost')
        # #plt.pause(0.01)
        # plt.draw

        if epoch % 1000 == 0:
            cost_WM = similarity_matching_cost(x_curr, model, C_target, alpha, beta_1, beta_2)
            print(f'Epoch {epoch}, Cost: {cost_WM.item()}')
            end_time=time.time()
            elapsed_time = end_time - start_time
            print(f"Iteration {i+1}: {elapsed_time:.6f} seconds")



    for inn in range(Yt_WM.shape[2]):
        selYInx = inn#100  # np.random.choice(range(num_sel), 1, replace=False)
        y_WM_np = Yt_WM[:,100:-200,selYInx]
        Ds_v[inn] = 0#compute_diffusion_constants(y_WM_np, plot=False)
        #volume_v[inn] = estimate_volume_convex_hull(y_WM_np.T)
        volume_v[inn] = compute_entropy_from_histogram(y_WM_np)
        Similarity[inn,:,:] = np.matmul(y_WM_np, y_WM_np.T)

    Ds = np.mean(Ds_v,axis=0)
    entrop = np.mean(volume_v,axis=0)

    return Ds, entrop, Similarity, Yt_WM, model
  
##############################################################################
##############################################################################
#############################  MAIN PART  ####################################
##############################################################################
##############################################################################

input_dim = 5#3  # Example input dimension
output_dim = 6  # Example output dimension
tot_iter = 10000  # Maximum iterations
dt = 0.05
lr = 0.1
num_samples = 10000
stdW = 0
stdM = 0
alpha = 1
beta_1 = 0.005
beta_2 = 0.005
num_copies = 1

model = PlaceCellNetwork(num_copies,input_dim, output_dim, tot_iter, dt, alpha, beta_1, beta_2)
X = torch.randn(num_samples, input_dim-1)  # Example input data
binary_variable = torch.randint(0, 2, (num_samples, 1))
X = torch.cat((X, binary_variable), dim=1)
Y = model(X)  # Apply the forward pass
rho = 0.0
batch_size = 500

# C_target_np = np.full((output_dim, output_dim), rho)
# noise = np.random.normal(0, 0.01, size=(output_dim, output_dim))
# noise[np.arange(output_dim), np.arange(output_dim)] = 0  # Zero out diagona l noise
# C_target_np += noise
# np.fill_diagonal(C_target_np, 1)
# C_target = torch.tensor(C_target_np, dtype=torch.float32)
# C_target[~torch.eye(output_dim, dtype=bool)] *= -1  # Only modify non-diagonal elements
# upper_tri_A = torch.triu(C_target)
# C_target = upper_tri_A + upper_tri_A.t() - torch.diag(torch.diag(upper_tri_A))

Ds0, entropy0, Similarity0, Yt_WM0, model_WM0 =  Simulate_Drift_NL(X, stdW, stdM, rho, batch_size, model, input_dim,output_dim, lr, alpha, beta_1, beta_2)

# M = model_WM0.M
# C = C_target
# M_diag = torch.diag(M)
# M_ii = M_diag.unsqueeze(1) 
# M_jj = M_diag.unsqueeze(0) 
# E = torch.zeros_like(M)
# mask = torch.eye(M.size(0), dtype=torch.bool)
# E[~mask] = (M[~mask] / (M_ii * M_jj)[~mask]) - (C[~mask] / (torch.sqrt(M_ii) * torch.sqrt(M_jj))[~mask])


#sel_inx=100
#y=model_WM0(X[sel_inx,:])
#yx =y.detach()
#Yt_WM = yx.t()
#selYInx = 100  # np.random.choice(range(num_sel), 1, replace=False)
#y_WM_np = Yt_WM0[:,:,sel_inx]
# Assuming y is the output from the last epoch
#for i in range(output_dim):
#    plt.plot(y_WM_np[i , ::20], label=f'Output dimension {i+1}', alpha=0.6)
#plt.show()

#eigenvalues = [4.5, 3.5, 1]  + [0.01] * (input_dim - 3)  # Eigenvalues for the covariance matrix
#X = generate_PSP_input_torch(eigenvalues, input_dim, num_samples)

#model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim)
#Ds0, volume0, y_WM_np0, model_WM0 =  Simulate_Drift(X, 0, 0, 0, auto, model_WM)
#avg_Ds0 = np.mean(Ds0)
#print(f"stdW: {0:.2f}, stdM: {0:.2f}, rho: {0:.2f}, Avg Ds: {np.mean(avg_Ds0):.4f}, Volume: {np.mean(volume0):.4f}")

stdWs = np.linspace(0.02, 0.05, 5)
stdMs = np.linspace(0, 0.05 , 5)
rhos = np.linspace(-0.1, 0.1, 3)
# Prepare to store the results
#Ds_results = np.zeros((len(stdWs), len(stdMs), len(rhos)))
Ds_results = np.zeros((len(stdWs), len(stdMs)))
entropy_results = np.zeros_like(Ds_results)
Similarity_results = np.zeros((len(stdWs), len(stdMs), Similarity0.shape[0], Similarity0.shape[1], Similarity0.shape[2]))
        
row_to_replicate = X[99, :].unsqueeze(0)  # Get the 100th row (index 99) and add a new dimension to make it [1, 5]
X_rep = row_to_replicate.repeat(10000, 1)  # 
y_WM_v = np.zeros((100,10,800))

for i, stdW in enumerate(stdWs):
#for j, stdM in enumerate(stdMs):
    #for k, rho in enumerate(rhos):
        # if k == 0 and i == 0 and j == 0:
        #     Ds0, entropy0, Yt_WM0, model_WM0 =  Simulate_Drift_NL(X, 0, 0, rho, auto, model, input_dim, output_dim, dt, alpha, beta_1, beta_2)
        #     Yt_WM_storage = np.zeros((len(stdWs), len(stdMs), len(rhos), Yt_WM0.shape[0], Yt_WM0.shape[1], Yt_WM0.shape[2]))
    #if i == 0 and j == 0:
    j = 0

    Ds0, entropy0, Similarity0, Yt_WM0, model_WM0 =  Simulate_Drift_NL(X, stdW, stdM, rho, batch_size, model, input_dim, output_dim, dt, alpha, beta_1, beta_2)
        #Yt_WM_storage[i, j, k] = Yt_WM0#np.zeros((len(stdWs), len(stdMs), len(rhos), Yt_WM0.shape[0], Yt_WM0.shape[1], Yt_WM0.shape[2]))
    k = 0
    rho = 0.0
    Ds, ent, Simil, Yt_WM, model =  Simulate_Drift_NL(X, stdW, stdM, rho, batch_size, model_WM0, input_dim, output_dim, dt, alpha, beta_1, beta_2)
    
    for h in range(100):
        Ds, ent, Simil, Yt_WM, model_rep =  Simulate_Drift_NL(X_rep, stdW, stdM, rho, 1, model_WM0, input_dim, output_dim, dt, alpha, beta_1, beta_2)
        y_WM_v[h,:,:] = Yt_WM[:,:-200,1]


    avg_Ds = np.mean(Ds)
    Ds_results[i, j] = avg_Ds
    entropy_results[i, j] = ent
    #Similarity_results[i, j,:,:,:] = Simil
    print(f"stdW: {stdW:.2f}, stdM: {stdM:.2f}, rho: {rho:.2f}, Avg Ds: {avg_Ds:.4f}, entropy: {ent:.4f}")
    #Yt_WM_storage[i, j, k] = Yt_WM

    sel_inx=1
    y_WM_np = Yt_WM[:,:-200,sel_inx]
    for ii in range(output_dim):
        plt.plot(y_WM_np[ii , ::20], label=f'Output dimension {ii+1}', alpha=0.6)
    plt.show()

if 1==2:

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
