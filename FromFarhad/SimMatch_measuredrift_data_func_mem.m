function [corefile] = SimMatch_measuredrift_data_func_mem_hs(params)
if nargin < 1
    params = [];
end
if isfield(params,'n'); n=params.n; else n=10; end
if isfield(params,'m'); m=params.m; else m=20; end
if isfield(params,'Nparticles'); Nparticles=params.Nparticles; else Nparticles=6; end
if isfield(params,'synnoise'); synnoise=params.synnoise; else synnoise=1*1e-2; end
if isfield(params,'eta'); eta=params.eta; else eta=0.01; end
if isfield(params,'numtrialstims'); numtrialstims = params.numtrialstims; else numtrialstims = min(m,n); end
if isfield(params,'sigmaperp'); sigmaperp=params.sigmaperp; else sigmaperp = 0.05; end
if isfield(params,'sigmapar'); sigmapar=params.sigmapar; else sigmapar = 1; end
if isfield(params,'sigmas'); sigmas=reshape(params.sigmas,length(params.sigmas),1); else sigmas = [sigmapar*ones(min(m,n),1);sigmaperp*ones(abs(n-m),1)]; end
if isfield(params,'overwrite'); overwrite=params.overwrite; else overwrite=1; end
if isfield(params,'outputfolder'); outputfolder=params.outputfolder; else outputfolder=''; end
if isfield(params,'mode'); mode=params.mode; else mode='synaptic'; end
if isfield(params,'stdZ'); stdZ=params.stdZ; else stdZ=synnoise; end
if isfield(params,'T'); T=params.T; else T=1e4; end
if isfield(params,'Tpretrain'); Tpretrain=params.Tpretrain; else Tpretrain=1e5; end
if isfield(params,'runPretrain'); runPretrain=params.runPretrain; else runPretrain=1; end
if isfield(params,'plotflag'); plotflag = params.plotflag; else plotflag = 1; end
if isfield(params,'datatype'); datatype=params.datatype; else datatype='gaussian'; end
if isfield(params,'npool'); npool=params.npool; else npool=8; end
if isfield(params,'rho_noise'); rho_noise=params.rho_noise; else rho_noise=0; end
if isfield(params,'excit_corr'); excit_corr=params.excit_corr; else excit_corr=1; end
if isfield(params,'excit_rho'); excit_rho=params.excit_rho; else excit_rho=[]; end
if isfield(params,'eig_decay'); eig_decay=params.eig_decay; else eig_decay=[]; end
if isfield(params,'D_eval'); D_eval = params.D_eval; else D_eval = [1 2 5 10 20 50 100]; end
if isfield(params,'plotfolder'); plotfolder = params.plotfolder; else plotfolder = ''; end
if isfield(params,'plotprefix'); plotprefix = params.plotprefix; else plotprefix = ''; end
if isfield(params,'seed'); seed=params.seed; else seed=[]; end

corefile = ['ResultSimMatch_',datatype,'_',num2str(n),'_',num2str(m),'_',num2str(Nparticles),'_synnoise_',num2str(synnoise),'_eta',num2str(eta),'_mode_',mode,'_rho_',num2str(rho_noise)];
if ~isempty(eig_decay)
    corefile = [corefile,'_eig_',num2str(eig_decay)];
end
if ~isempty(seed)
    corefile = [corefile,'_seed_',num2str(seed)];
end
filename = [outputfolder,corefile,'.mat'];
% Ensure output and plot directories exist
if ~isempty(outputfolder) && ~exist(outputfolder,'dir'); mkdir(outputfolder); end
% set unique plot folder per run if not provided
if ~exist('plot','dir'); mkdir('plot'); end
if isempty(plotfolder)
    tstamp = datestr(now,'yyyymmdd_HHMMSS');
    plotfolder = fullfile('plot', [mode '_n' num2str(n) '_m' num2str(m) '_rho_' num2str(rho_noise) '_' tstamp]);
end
if ~exist(plotfolder,'dir'); mkdir(plotfolder); end
if isempty(plotprefix)
    pf = '';
else
    pf = [plotprefix '_'];
end
if isfile(filename) && overwrite == 0
    disp(['File Exists. Skipping.'])
    return
end
tic
% Optional RNG seed for reproducibility
if ~isempty(seed)
    rng(seed,'twister');
end
input_dim = n; output_dim = m;
num_sample = 1e4;
% Build input covariance eigenvalues
if ~isempty(eig_decay)
    % Power-law eigenvalue spectrum: lambda_i ~ i^{-eig_decay}
    idxs = (1:input_dim)';
    base_sig = idxs.^(-eig_decay/2);
    % Normalize to keep leading variance comparable to sigmapar if provided
    if ~exist('sigmapar','var') || isempty(sigmapar)
        sigmapar = 1;
    end
    base_sig = sigmapar * base_sig / base_sig(1);
    sigmas_eff = base_sig;
else
    sigmas_eff = sigmas(1:input_dim);
end
D = sigmas_eff.^2;
[Q, ~] = qr(randn(n)); Sigma = Q * diag(D) * Q'; X = mvnrnd(zeros(1, n), Sigma, num_sample)';
[V,~,~] = svd(X*X');
V = V(:,1:min(numtrialstims,output_dim));
trialstims = V(:,1:numtrialstims);
Vnorm = norm(V*V','fro');
W = randn(output_dim,input_dim);
M = eye(output_dim);
tot_inital = Tpretrain;
learnRate = eta;
dt = learnRate;
stdW = synnoise;
stdM = synnoise;
% Build correlated noise Cholesky across output neurons
rho_min = -1/(output_dim-1) + 1e-6;
rho_eff = min(max(rho_noise,rho_min),0.99);
Cnoise = (1-rho_eff)*eye(output_dim) + rho_eff*ones(output_dim);
try
    Lnoise = chol(Cnoise,'lower');
catch
    Lnoise = chol(Cnoise + 1e-8*eye(output_dim),'lower');
end
% Excitability noise correlation (can differ from synaptic rho)
if isempty(excit_rho)
    rho_ex = rho_eff; % match synaptic by default
else
    rho_ex = min(max(excit_rho,rho_min),0.99);
end
Cnoise_ex = (1-rho_ex)*eye(output_dim) + rho_ex*ones(output_dim);
try
    Lnoise_ex = chol(Cnoise_ex,'lower');
catch
    Lnoise_ex = chol(Cnoise_ex + 1e-8*eye(output_dim),'lower');
end
if runPretrain == 1
    Ms = zeros(output_dim,output_dim,tot_inital);
    Ws = zeros(output_dim,input_dim,tot_inital);
    initial_pspErr = nan(tot_inital,1);
    for i = 1:tot_inital
        curr_inx = randperm(num_sample,1);
        Y = pinv(M)*W*X(:,curr_inx);
        noiseW = sqrt(dt)*stdW*(Lnoise*randn(output_dim,input_dim));
        noiseM = sqrt(dt)*stdM*(Lnoise*randn(output_dim,output_dim));
        noiseM = (noiseM + noiseM')/2; % symmetrize
        W = (1-dt)*W + dt*Y*X(:,curr_inx)' + noiseW;
        M = (1-dt)*M + dt*Y*Y' + noiseM;
        Ms(:,:,i) = M; Ws(:,:,i) = W;
        F = pinv(M)*W;
        initial_pspErr(i) = norm(F'*F - V(:,1:min(n,output_dim))*V(:,1:min(n,output_dim))','fro')/Vnorm;
    end
    if plotflag == 1
        fig = figure;
        subplot(1,2,1); plot(1:length(initial_pspErr),initial_pspErr,'LineWidth',2);
        subplot(1,2,2); plot(length(initial_pspErr)/2:length(initial_pspErr),initial_pspErr(end/2:end),'LineWidth',2);
        sgtitle('Pretraining Subspace Loss');
        set(fig,'Position',[1,100,800,400]);
        saveas(fig, fullfile(plotfolder, 'pretraining_loss.png'));
    end
    Mhat = mean(Ms(:,:,end-100:end),3);
    What = mean(Ws(:,:,end-100:end),3);
    W0 = What; M0 = Mhat;
else
    F = pinv(M)*W;
    initial_pspErr = norm(F'*F - V(:,1:output_dim)*V(:,1:output_dim)','fro')/Vnorm;
end
% Short-term statistics
T1 = Tpretrain-1000; T2 = Tpretrain;
Hs = zeros(m,T2 - T1 + 1,numtrialstims);
for t = T1:T2
    Wi = Ws(:,:,t); Mi = Ms(:,:,t);
    Hs(:,t-T1+1,:) = inv(Mi)*Wi*trialstims;
end
NeuronsCorrsPerStim = zeros(m,m,numtrialstims);
for st = 1:numtrialstims
    NeuronsCorrsPerStim(:,:,st) = corrcoef(squeeze(Hs(:,:,st)'));
end
AveNoiseCorr = mean(NeuronsCorrsPerStim,3);
Hsave = squeeze(mean(Hs,2));
SignalCorr = corr(Hsave');
% Drift measurement
Nchunks = ceil(Nparticles / npool);
Rautomean = zeros(T,numtrialstims);
flucs = zeros(1,numtrialstims);
% Add for point 3: pair drift metrics vs correlations
pair_drift_vars = zeros(m*(m-1)/2,1);
pair_abs_drift_vars = zeros(m*(m-1)/2,1);
pair_abs_drift_meanabs = zeros(m*(m-1)/2,1); % E[ | |h_i| - |h_j| | ]
% increment-based metrics
pair_drift_incr_vars = zeros(m*(m-1)/2,1);
pair_abs_drift_incr_meanabs = zeros(m*(m-1)/2,1);
pair_signal_corrs = zeros(m*(m-1)/2,1);
pair_noise_corrs = zeros(m*(m-1)/2,1);
pair_idx = 1;
for ich = 1:Nchunks
    Rauto_chunk= zeros(npool,numtrialstims,T);
    Hsnorms_chunk = zeros(npool, T, numtrialstims);
    hs_ = zeros(npool,m,T,numtrialstims);
    % Prediction error vs lag D (GLM generalization drift)
    ND = length(D_eval);
    pred_mse_sum_chunk = zeros(ND,1);
    pred_mse_count_chunk = zeros(ND,1);
    pred_mse_neuron_sum_chunk = zeros(m, ND);
    for ip = 1:npool
        W = W0; M = M0; h0 = inv(M0)*W*trialstims;
        hs = zeros(m,T,numtrialstims);
        zeta = ones(output_dim,1); % for excitability
        % ring buffers for previous operators
        maxD = 0; if ND>0, maxD = max(D_eval); end
        if maxD>0
            Bbuf = zeros(m, input_dim, maxD);
        end
        for i = 1:T
            curr_inx = randperm(num_sample,1);
            if strcmp(mode,'synaptic')
                Y = pinv(M)*W*X(:,curr_inx);
                noiseW = sqrt(dt)*stdW*(Lnoise*randn(output_dim,input_dim));
                noiseM = sqrt(dt)*stdM*(Lnoise*randn(output_dim,output_dim));
                noiseM = (noiseM + noiseM')/2;
                W = (1-dt)*W + dt*Y*X(:,curr_inx)' + noiseW;
                M = (1-dt)*M + dt*Y*Y' + noiseM;
            else % excitability
                ff = W*X(:,curr_inx);
                ff_mod = zeta .* ff;
                Y = pinv(M)*ff_mod;
                % excitability noise across neurons; optionally correlated
                if excit_corr == 1
                    zeta = zeta + sqrt(dt)*stdZ*(Lnoise_ex*randn(output_dim,1));
                else
                    zeta = zeta + sqrt(dt)*stdZ*randn(output_dim,1);
                end
            end
            % build operator B_op = inv(M) * diag(zeta) * W for prediction
            if strcmp(mode,'synaptic')
                B_op = pinv(M)*W;
            else
                B_op = pinv(M)*(bsxfun(@times, zeta, W));
            end
            % evaluate prediction errors BEFORE writing current operator to ring buffer
            if maxD>0
                k = mod(i-1, maxD) + 1;
                for d = 1:ND
                    Dlag = D_eval(d);
                    if i - Dlag >= 1
                        kp = mod(i - Dlag - 1, maxD) + 1;
                        B_prev = Bbuf(:,:,kp);
                        x_now = X(:,curr_inx);
                        y_pred = B_prev * x_now;
                        e = Y - y_pred;
                        pred_mse_sum_chunk(d) = pred_mse_sum_chunk(d) + mean(e.^2);
                        pred_mse_neuron_sum_chunk(:,d) = pred_mse_neuron_sum_chunk(:,d) + e.^2;
                        pred_mse_count_chunk(d) = pred_mse_count_chunk(d) + 1;
                    end
                end
                % now write current operator to buffer slot for future lags
                Bbuf(:,:,k) = B_op;
            end
            hs(:,i,:) = inv(M)*bsxfun(@times, zeta, W*trialstims); % general for both modes
        end
        for st = 1:numtrialstims
            hs0 = h0(:,st);
            hsc = squeeze(hs(:,:,st));
            Rauto_chunk(ip,st,:) = hs0'*hsc ./(norm(hs0)*sqrt(sum(hsc.^2,1)));
        end
        Hsnorms_chunk(ip,:,:) = sqrt(sum(hs.^2,1));
        hs_(ip,:,:,:) = hs;
    end
    Rautomean = Rautomean + squeeze(mean(Rauto_chunk,1))';
    flucs = flucs + mean(squeeze(var(Hsnorms_chunk,[],2)),1);
    % accumulate prediction error
    if ich == 1
        pred_mse_sum = zeros(ND,1);
        pred_mse_count = zeros(ND,1);
        pred_mse_neuron_sum = zeros(m, ND);
    end
    pred_mse_sum = pred_mse_sum + pred_mse_sum_chunk;
    pred_mse_count = pred_mse_count + pred_mse_count_chunk;
    pred_mse_neuron_sum = pred_mse_neuron_sum + pred_mse_neuron_sum_chunk;
    % Pair drift accumulation for this chunk
    hs_mean_chunk = squeeze(mean(hs_,1)); % m x T x numtrialstims
    pair_idx_local = 1;
    for st = 1:numtrialstims
        pair_idx_local = 1;
        for i = 1:m
            for j = i+1:m
                diff_var = var(hs_mean_chunk(i,:,st) - hs_mean_chunk(j,:,st));
                if ich == 1 && st == 1
                    % initialize on first use
                    if ~exist('pair_drift_vars_sum','var')
                        pair_drift_vars_sum = zeros(m*(m-1)/2,1);
                        pair_abs_drift_vars_sum = zeros(m*(m-1)/2,1);
                    end
                end
                pair_drift_vars_sum(pair_idx_local) = pair_drift_vars_sum(pair_idx_local) + diff_var;
                % absolute-magnitude difference variance and mean-abs
                abs_diff_var = var(abs(hs_mean_chunk(i,:,st)) - abs(hs_mean_chunk(j,:,st)));
                pair_abs_drift_vars_sum(pair_idx_local) = pair_abs_drift_vars_sum(pair_idx_local) + abs_diff_var;
                abs_diff_meanabs = mean(abs(abs(hs_mean_chunk(i,:,st)) - abs(hs_mean_chunk(j,:,st))));
                if ~exist('pair_abs_drift_meanabs_sum','var'); pair_abs_drift_meanabs_sum = zeros(m*(m-1)/2,1); end
                pair_abs_drift_meanabs_sum(pair_idx_local) = pair_abs_drift_meanabs_sum(pair_idx_local) + abs_diff_meanabs;
                % increment-based metrics
                dhi = diff(hs_mean_chunk(i,:,st));
                dhj = diff(hs_mean_chunk(j,:,st));
                incr_diff_var = var(dhi - dhj);
                if ~exist('pair_drift_incr_vars_sum','var'); pair_drift_incr_vars_sum = zeros(m*(m-1)/2,1); end
                pair_drift_incr_vars_sum(pair_idx_local) = pair_drift_incr_vars_sum(pair_idx_local) + incr_diff_var;
                incr_abs_meanabs = mean(abs(abs(dhi) - abs(dhj)));
                if ~exist('pair_abs_drift_incr_meanabs_sum','var'); pair_abs_drift_incr_meanabs_sum = zeros(m*(m-1)/2,1); end
                pair_abs_drift_incr_meanabs_sum(pair_idx_local) = pair_abs_drift_incr_meanabs_sum(pair_idx_local) + incr_abs_meanabs;
                pair_idx_local = pair_idx_local + 1;
            end
        end
    end
end
Rautomean = Rautomean/Nchunks; flucs = flucs/Nchunks;
% Finalize pair drift averaged over st and chunks
pair_drift_vars = pair_drift_vars_sum / (Nchunks * numtrialstims);
pair_abs_drift_vars = pair_abs_drift_vars_sum / (Nchunks * numtrialstims);
if exist('pair_abs_drift_meanabs_sum','var')
    pair_abs_drift_meanabs = pair_abs_drift_meanabs_sum / (Nchunks * numtrialstims);
end
if exist('pair_drift_incr_vars_sum','var')
    pair_drift_incr_vars = pair_drift_incr_vars_sum / (Nchunks * numtrialstims);
end
if exist('pair_abs_drift_incr_meanabs_sum','var')
    pair_abs_drift_incr_meanabs = pair_abs_drift_incr_meanabs_sum / (Nchunks * numtrialstims);
end
% Fill pair correlation vectors once
pair_signal_corrs = zeros(m*(m-1)/2,1);
pair_noise_corrs = zeros(m*(m-1)/2,1);
pair_idx = 1;
for i = 1:m
    for j = i+1:m
        pair_signal_corrs(pair_idx) = SignalCorr(i,j);
        pair_noise_corrs(pair_idx) = AveNoiseCorr(i,j);
        pair_idx = pair_idx + 1;
    end
end
% Plot for point 3
if plotflag == 1
    fig_corr = figure; subplot(2,2,1); scatter(pair_signal_corrs, pair_drift_vars, 'filled'); xlabel('Signal Corr'); ylabel('Pair Drift Var');
    subplot(2,2,2); scatter(pair_noise_corrs, pair_drift_vars, 'filled'); xlabel('Noise Corr'); ylabel('Pair Drift Var');
    subplot(2,2,3); scatter(pair_noise_corrs, pair_abs_drift_meanabs, 'filled'); xlabel('Noise Corr'); ylabel('E[||h_i|-|h_j||]');
    subplot(2,2,4); scatter(pair_noise_corrs, pair_abs_drift_incr_meanabs, 'filled'); xlabel('Noise Corr'); ylabel('E[||Δh_i|-|Δh_j||]');
    sgtitle('Drift metrics vs correlations');
    saveas(fig_corr, fullfile(plotfolder, 'drift_vs_correlations.png'));
end
% Compute Ds
Rslopes = zeros(numtrialstims,3);
xi = 0:size(Rautomean,1)-1;
for st= 1:numtrialstims
    ry = -log(Rautomean(1:end,st));
    mdlr = fitlm(xi(1:end), real(ry));
    Rslopes(st,:) = [mdlr.Coefficients.Estimate(2),mdlr.Coefficients.SE(2),mdlr.Rsquared.Ordinary];
end
Ds = Rslopes(:,1);
% Additional visualizations
if plotflag == 1
    % Rauto curves across stimuli and mean
    figR = figure; hold on
    xiR = 0:size(Rautomean,1)-1;
    plot(xiR, Rautomean, 'Color', [0.7 0.7 0.7])
    plot(xiR, mean(Rautomean,2), 'k', 'LineWidth', 2)
    xlabel('time'); ylabel('Rauto'); title('Autocorrelation decay across stimuli','Interpreter','none')
    set(figR,'Position',[50,50,700,350]);
    saveas(figR, fullfile(plotfolder, 'rauto_curves.png'));
    % PSP projector similarity visualization
    try
        k = min(n, output_dim);
        Ptarget = V(:,1:k) * V(:,1:k)';
        F0 = pinv(M0)*W0; Plearn = F0'*F0;
        figP = figure('Position',[60,120,900,300]);
        subplot(1,3,1); imagesc(Ptarget); axis square; colorbar; title('Target projector','Interpreter','none')
        subplot(1,3,2); imagesc(Plearn); axis square; colorbar; title('Learned F^T F','Interpreter','none')
        subplot(1,3,3); imagesc(Plearn - Ptarget); axis square; colorbar; title('Difference','Interpreter','none')
        saveas(figP, fullfile(plotfolder, 'psp_similarity.png'));
    catch
    end
    % Per-neuron MSE drift slope distribution (if available)
    if exist('pred_mse_slope_neuron','var') && ~isempty(pred_mse_slope_neuron)
        figh = figure; histogram(pred_mse_slope_neuron, 20); xlabel('pred\\_mse slope per neuron'); ylabel('# neurons');
        title('Distribution of MSE drift slopes','Interpreter','none'); set(figh,'Position',[80,160,600,350]);
        saveas(figh, fullfile(plotfolder, 'mse_drift_slope_hist.png'));
    end
end
% finalize prediction error vs lag D
if exist('pred_mse_sum','var') && any(pred_mse_count>0)
    pred_mse = pred_mse_sum ./ max(pred_mse_count,1);
    pred_mse_neuron = pred_mse_neuron_sum ./ max(pred_mse_count',1);
    % Fit a single drift value from pred_mse vs D (linear slope)
    try
        xiD = D_eval(:);
        mdlm = fitlm(xiD, pred_mse(:));
        pred_mse_slope = mdlm.Coefficients.Estimate(2);
        pred_mse_slope_se = mdlm.Coefficients.SE(2);
        % Per-neuron slopes
        pred_mse_slope_neuron = nan(m,1);
        for jj = 1:m
            mdlmj = fitlm(xiD, pred_mse_neuron(jj,:)');
            pred_mse_slope_neuron(jj) = mdlmj.Coefficients.Estimate(2);
        end
    catch
        pred_mse_slope = NaN; pred_mse_slope_se = NaN; pred_mse_slope_neuron = nan(m,1);
    end
    % Plot MSE drift curves
    if plotflag == 1
        figm = figure; 
        subplot(1,2,1); plot(D_eval, pred_mse,'-o','LineWidth',2); xlabel('D (lag)'); ylabel('MSE using \beta(t-D) on x(t)'); title('Prediction error vs lag','Interpreter','none');
        hold on; if exist('pred_mse_slope','var') && ~isnan(pred_mse_slope); yfit = mdlm.Coefficients.Estimate(1) + pred_mse_slope*D_eval; plot(D_eval,yfit,'--','LineWidth',1.5); end
        subplot(1,2,2); 
        pred_mse_norm = pred_mse - pred_mse(1); plot(D_eval, pred_mse_norm,'-o','LineWidth',2); xlabel('D (lag)'); ylabel('MSE increase from D=0'); title(['Drift slope=', num2str(pred_mse_slope,3)],'Interpreter','none');
        set(figm,'Position',[50,200,900,350]);
        saveas(figm, fullfile(plotfolder, 'mse_drift.png'));
    end
else
    pred_mse = [];
    pred_mse_neuron = [];
    pred_mse_slope = NaN; pred_mse_slope_se = NaN; pred_mse_slope_neuron = [];
end
save(filename,'Ds','flucs','Rautomean','trialstims','params','T','initial_pspErr','sigmas','eta','Nparticles','pair_drift_vars','pair_abs_drift_vars','pair_abs_drift_meanabs','pair_drift_incr_vars','pair_abs_drift_incr_meanabs','pair_signal_corrs','pair_noise_corrs','rho_noise','eig_decay','seed','D_eval','pred_mse','pred_mse_neuron','pred_mse_slope','pred_mse_slope_se','pred_mse_slope_neuron');
disp('Finished. Results saved.')
toc
end