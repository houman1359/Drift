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
import pdb

# O2 branch v426

# Check if CUDA is available and set the default tensor type
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA on GPU")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

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
        #Y = Y.detach().numpy()
        Y = Y.cpu().detach().numpy()  # Ensure tensor is on CPU and converted to numpy

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
        # Ensure the tensor is moved to the CPU and converted to NumPy before processing with np.histogram
        hist, _ = np.histogram(distances[i, :].cpu().numpy(), bins=bins, density=True)
        prob_dist = hist / np.sum(hist)
        y[i] = entropy(prob_dist)

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
    def __init__(self, input_dim, output_dim, MaxIter, dt, device, alpha=0.0, lbd1=0.0, lbd2=0.0):
        super(PlaceCellNetwork, self).__init__()
        self.device = device  # Define the device where the parameters will be stored
        self.W = nn.Parameter(torch.randn(output_dim, input_dim).to(device))
        self.M = nn.Parameter(torch.eye(output_dim).to(device))
        self.b = nn.Parameter(torch.zeros(output_dim).to(device))
        self.MaxIter = MaxIter
        self.dt = dt
        self.alpha = alpha
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        self.errTrack = torch.rand(5, 1).to(device)  # Store on the correct device

    def forward(self, X):
        X = X.to(self.device)  # Ensure input is on the same device as model parameters
        batch_size = X.size(0)
        Y = torch.zeros(batch_size, self.W.size(0), device=self.device)
        Yold = Y.clone()
        diag_M = torch.diag_embed(torch.diag(self.M))  # Correct use of torch.diag_embed
        Wx = torch.mm(X, self.W.t())  # Use matrix multiplication correctly
        MO = self.M - diag_M

        for count in range(self.MaxIter):
            M_Y = torch.mm(Yold, MO)
            dt = self.dt  # Time step might be adaptive based on some criterion, simplifying here
            du = -Yold + Wx - np.sqrt(self.alpha) * self.b - M_Y
            uy = Y + dt * du
            Y = torch.maximum(uy - self.lbd1, torch.zeros_like(uy)) / (self.lbd2 + torch.diag(self.M).unsqueeze(0))

            err = torch.norm(Y - Yold) / (torch.norm(Yold) + 1e-10)
            if err < 1e-4:
                break
            Yold = Y.clone()

        return Y

class PlaceCellNetworkold(nn.Module):
    def __init__(self, input_dim, output_dim, MaxIter, dt, alpha=0.0, lbd1=0.0, lbd2=0.0):
        super(PlaceCellNetwork, self).__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim, device=device))
        self.M = nn.Parameter(torch.eye(output_dim, device=device))
        self.b = nn.Parameter(torch.zeros(output_dim, device=device))
        self.MaxIter = MaxIter
        self.dt = dt
        self.alpha = alpha
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        self.errTrack = torch.rand(5, 1, device=device)  # Store on the correct device

    def forward(self, X):
        X = X.to(self.W.device)  # Ensure input is on the same device as model parameters
        batch_size = X.size(0)
        Y = torch.zeros(batch_size, self.W.size(0), device=self.W.device)
        Yold = Y.clone()
        diag_M = torch.diag(self.M)
        Wx = torch.mm(X, self.W.t())  # Use matrix multiplication correctly
        MO = self.M.t() - diag_M

        for count in range(self.MaxIter):
            #M_Y = torch.mm(Yold, self.M.t())  # Ensure proper matrix multiplication
            M_Y = torch.mm(Yold, MO)  # Ensure proper matrix multiplication
            dt = self.dt#max(self.dt / (1 + count / 10), 1e-3)
            du = -Yold + Wx - np.sqrt(self.alpha) * self.b - M_Y
            Y += dt * du  # Updated incrementally
            Y = torch.maximum(Y - self.lbd1, torch.zeros_like(Y)) / (self.lbd2 + diag_M)

            err = torch.norm(Y - Yold) / (torch.norm(Yold) + 1e-10) / dt
            if err.item() < 1e-4:  # Use item() to extract the scalar value
                break
            Yold = Y.clone()

        return Y

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

def Simulate_Drift_NL(X, stdW , stdM, rho, auto, model, input_dim, output_dim, lr, alpha=0.0, beta_1=0.0, beta_2=0.0):

    X = X.to(device)
    model.to(device)

    plt.ion()  # Turn on interactive plotting
    fig, ax = plt.subplots()  # Create a figure and axes object

    #stdW = syn_noise_std
    #stdM = syn_noise_std
    num_sel = 200 # randomly selected samples used to calculate the drift and diffusion constants
    step = 10#10    # store every 10 updates
    time_points = round(tot_iter / step)
    #sel_inx = np.random.permutation(num_samples)[:num_sel].to(device)
    sel_inx = torch.randperm(num_samples)[:num_sel].to(device)

    Yt_WM = torch.zeros(output_dim, time_points, num_sel, device=device)
    Ds_v = torch.zeros(time_points, device=device)
    volume_v = torch.zeros(time_points, device=device)
    Similarity = torch.zeros(time_points, output_dim, output_dim, device=device)

    #rho = 0.0
    #C_target_np = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])  # Target correlation matrix
    #C_target = torch.tensor(C_target_np, dtype=torch.float32)
    C_target_np = torch.full((output_dim, output_dim), rho, device=device)
    noise = torch.randn(output_dim, output_dim, device=device) * 0.01
    noise.fill_diagonal_(0)  # Zero out diagonal noise
    C_target_np += noise
    C_target_np.fill_diagonal_(1)
    C_target = C_target_np.triu() + C_target_np.triu(1).transpose(0, 1) - torch.diag(torch.diag(C_target_np))

    #model_WM = SimilarityMatchingNetwork_WM(input_dim, output_dim)
    optimizer_WM = torch.optim.SGD(model.parameters(), lr=lr)
    DeltaWM_W_manual = torch.nn.Parameter(torch.randn(output_dim, input_dim).to(device))
    DeltaWM_M_manual = torch.nn.Parameter(torch.eye(output_dim).to(device))
    nn = 0

    #fig, ax = plt.subplots()    
    start_time = time.time()
    for epoch in range(tot_iter):  # Number of epochs

        if epoch % 1000 ==1:
            start_time = time.time()
            #ax.clear()  # Clear the axes to update the plot
            #ax.plot(model.W[:,1].detach().cpu().numpy(), model.W[:,2].detach().cpu().numpy(), 'b.')  # 'b.' plots a blue dot
            #plt.draw()  # Redraw the current figure
            #plt.pause(1)  # Pause for a bit to see the update
            #pdb.set_trace()  # Execution will pause here

        # Randomly select one sample
        curr_inx = torch.randint(0, num_samples, (100,), device=device) #torch.tensor([1])
        x_curr = X[curr_inx,:].to(device)  # Current input sample
        Y_WM = model(x_curr)

        xis = torch.randn_like(model.W) * stdW
        zetas = torch.randn_like(model.M) * stdM
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
        
        #DeltaW = lr * torch.mm(x_curr.t(), model(x_curr) - model.W) + torch.sqrt(torch.tensor(lr)) * xis
        DeltaW = lr * (torch.matmul(Y_WM.t(), x_curr) / x_curr.size(0) - model.W) + torch.sqrt(torch.tensor(lr)) * xis
        DeltaM = lr * (torch.matmul(Y_WM.t(), Y_WM) / Y_WM.size(0) - model.M) + torch.sqrt(torch.tensor(lr)) * zetas
        Deltab = lr * (np.sqrt(alpha) * torch.mean(Y_WM, dim=0) - model.b) + torch.sqrt(torch.tensor(lr)) * xi_b
            
        #pdb.set_trace()  # Execution will pause here

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
            else:
                print(DeltaM)
                #print(DeltaM)
                #print(Deltab)

        if epoch % step == 0: #and epoch>1000:
            y=model(X[sel_inx,:])
            yx =y.detach()
            Yt_WM[:,nn,:] = yx.t()
            nn += 1

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
            print(f"Iteration {epoch}: {elapsed_time:.6f} seconds")


    for inn in range(Yt_WM.shape[2]):
        selYInx = inn#100  # np.random.choice(range(num_sel), 1, replace=False)
        y_WM_np = Yt_WM[:,100:-200,selYInx]
        Ds_v[inn] = 0#compute_diffusion_constants(y_WM_np, plot=False)
        #volume_v[inn] = estimate_volume_convex_hull(y_WM_np.T)
        volume_v[inn] = compute_entropy_from_histogram(y_WM_np)
        y_WM_np_cpu = y_WM_np.cpu().numpy()
        similarity_matrix_np = np.matmul(y_WM_np_cpu, y_WM_np_cpu.T)
        similarity_matrix_tensor = torch.from_numpy(similarity_matrix_np).to(device)  # Convert numpy array to tensor and move to GPU

        Similarity[inn, :, :] = similarity_matrix_tensor  # Assign the tensor, which is now on the correct device

    #Ds = np.mean(Ds_v,axis=0)
    #Ds = Ds_v.mean().item()  # Compute the mean using PyTorch and convert to Python scalar
    #entrop = np.mean(volume_v,axis=0)
    Ds_mean = Ds_v.mean()
    volume_mean = volume_v.mean()

    #return Ds, entrop, Similarity, Yt_WM, model
    return Ds_mean, volume_mean, Similarity, Yt_WM, model

##############################################################################
##############################################################################
#############################  MAIN PART  ####################################
##############################################################################
##############################################################################

input_dim = 5#3  # Example input dimension
output_dim = 10  # Example output dimension
tot_iter = 10000  # Maximum iterations
dt = 0.05
lr = 0.01
num_samples = 10000
stdW = 0
stdM = 0
alpha = 1
beta_1 = 0.005
beta_2 = 0.005
model = PlaceCellNetwork(input_dim, output_dim, tot_iter, dt, alpha, beta_1, beta_2)
X = torch.randn(num_samples, input_dim-1, device=device)  # Example input data
X[:,0]+=1
X[:,1]+=2

binary_variable = torch.randint(0, 2, (num_samples, 1), device=device)
X = torch.cat((X, binary_variable), dim=1)
Y = model(X)  # Apply the forward pass
auto = 0
rho = 0.0

# C_target_np = np.full((output_dim, output_dim), rho)
# noise = np.random.normal(0, 0.01, size=(output_dim, output_dim))
# noise[np.arange(output_dim), np.arange(output_dim)] = 0  # Zero out diagona l noise
# C_target_np += noise
# np.fill_diagonal(C_target_np, 1)
# C_target = torch.tensor(C_target_np, dtype=torch.float32)
# C_target[~torch.eye(output_dim, dtype=bool)] *= -1  # Only modify non-diagonal elements
# upper_tri_A = torch.triu(C_target)
# C_target = upper_tri_A + upper_tri_A.t() - torch.diag(torch.diag(upper_tri_A))


 
Ds0, entropy0, Similarity0, Yt_WM0, model_WM0 =  Simulate_Drift_NL(X, stdW, stdM, rho, auto, model, input_dim,output_dim, lr, alpha, beta_1, beta_2)

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


stdWs = torch.linspace(0, 0.00005, 5, device=device)
stdMs = torch.linspace(0, 0.00005, 5, device=device)
rhos = torch.linspace(-0.1, 0.1, 3, device=device)


# Prepare to store the results
#Ds_results = np.zeros((len(stdWs), len(stdMs), len(rhos)))
Ds_results = torch.zeros((len(stdWs), len(stdMs)), device=device)
entropy_results = torch.zeros_like(Ds_results)
Similarity_results = torch.zeros((len(stdWs), len(stdMs), Similarity0.shape[0], Similarity0.shape[1], Similarity0.shape[2]), device=device)

for i, stdW in enumerate(stdWs):
    for j, stdM in enumerate(stdMs):
        #for k, rho in enumerate(rhos):
            # if k == 0 and i == 0 and j == 0:
            #     Ds0, entropy0, Yt_WM0, model_WM0 =  Simulate_Drift_NL(X, 0, 0, rho, auto, model, input_dim, output_dim, dt, alpha, beta_1, beta_2)
            #     Yt_WM_storage = np.zeros((len(stdWs), len(stdMs), len(rhos), Yt_WM0.shape[0], Yt_WM0.shape[1], Yt_WM0.shape[2]))
        #if i == 0 and j == 0:
        Ds0, entropy0, Similarity0, Yt_WM0, model_WM0 =  Simulate_Drift_NL(X, stdW, stdM, rho, auto, model, input_dim, output_dim, dt, alpha, beta_1, beta_2)
            #Yt_WM_storage[i, j, k] = Yt_WM0#np.zeros((len(stdWs), len(stdMs), len(rhos), Yt_WM0.shape[0], Yt_WM0.shape[1], Yt_WM0.shape[2]))
        k = 0
        rho = 0.0
        Ds, ent, Simil, Yt_WM, model =  Simulate_Drift_NL(X, stdW, stdM, rho, auto, model_WM0, input_dim, output_dim, dt, alpha, beta_1, beta_2)

        avg_Ds = Ds
        Ds_results[i, j] = avg_Ds
        entropy_results[i, j] = ent
        Similarity_results[i, j,:,:,:] = Simil
        print(f"stdW: {stdW:.2f}, stdM: {stdM:.2f}, rho: {rho:.2f}, Avg Ds: {avg_Ds:.4f}, entropy: {ent:.4f}")
        #Yt_WM_storage[i, j, k] = Yt_WM

        sel_inx = 100  # Ensure sel_inx is within the bounds of Yt_WM's dimensions
        if Yt_WM.size(2) > sel_inx:
            y_WM_np = Yt_WM[:, :, sel_inx].cpu().numpy()  # Move to CPU for plotting
            for ii in range(output_dim):
                plt.plot(y_WM_np[ii, ::20], label=f'Output dimension {ii+1}', alpha=0.6)
            plt.legend()
            plt.show(block=False)



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