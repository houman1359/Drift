% Eigenspectrum decay sweep (complexity dependence)
% PARAMETERS (edit here)
%   n, m                 : input/output dims
%   outputfolder         : results directory
%   synnoise, Nparticles : noise std and #particles
%   eta                  : learning rate / dt
%   overwrite, npool     : IO and parallelism controls
%   eig_decays           : vector of power-law exponents (lambda_i ~ i^{-eig_decay})
%   modes                : {'synaptic','excitability'}
%   rho_noise            : synaptic equicorr coefficient
%   Nreps, base_seed     : repeats and base seed
% Notes:
%   - To use heterogeneous/custom correlations, also set:
%       params.noise_corr_mode / params.Cnoise / params.alpha_range
%       params.excit_corr_mode / params.Cnoise_ex / params.alpha_ex_range
clear all
close all

outputfolder = 'Results/';
if ~exist(outputfolder,'dir'); mkdir(outputfolder); end
if ~exist('plot','dir'); mkdir('plot'); end
% One plot root for the entire sweep
tstamp = datestr(now,'yyyymmdd_HHMMSS');
plot_root = fullfile('plot', ['eigdecay_' num2str(n) 'x' num2str(m) '_' tstamp]);
if ~exist(plot_root,'dir'); mkdir(plot_root); end

% Base params
n = 12; m = 12;
synnoise = 1e-2;
Nparticles = 32;
eta = 0.001;
overwrite = 1;
npool = 8;

% Power-law decay exponents to test (lambda_i ~ i^{-eig_decay})
eig_decays = [0, 0.5, 1, 1.5, 2];
modes = {'synaptic','excitability'};
rho_noise = 0.2; % moderate correlation for synaptic; excitability can still use same param
Nreps = 3; base_seed = 20250;

for mi = 1:length(modes)
    mode = modes{mi};
    Ds_mean = nan(length(eig_decays),1);
    Ds_sem  = nan(length(eig_decays),1);
    pair_mean = nan(length(eig_decays),1);
    pair_sem  = nan(length(eig_decays),1);
    pair_abs_mean = nan(length(eig_decays),1);
    pair_abs_sem  = nan(length(eig_decays),1);
    for ei = 1:length(eig_decays)
        e = eig_decays(ei);
        Ds_reps = nan(Nreps,1);
        pair_reps = nan(Nreps,1);
        pair_abs_reps = nan(Nreps,1);
        pair_incr_reps = nan(Nreps,1);
        pair_abs_incr_reps = nan(Nreps,1);
        pred_mse_slope_reps = nan(Nreps,1);
        for rep = 1:Nreps
            params = [];
            params.n = n; params.m = m;
            params.Nparticles = Nparticles; params.synnoise = synnoise;
            params.eta = eta; params.numtrialstims = min(m,n);
            params.npool = npool; params.overwrite = overwrite;
            params.outputfolder = outputfolder;
            params.mode = mode; params.stdZ = synnoise;
            params.rho_noise = rho_noise; params.plotflag = (ei == ceil(length(eig_decays)/2) && rep == 1 && mi == 1);
            params.plotfolder = plot_root;
            params.eig_decay = e;
            params.seed = base_seed + 10000*ei + 100000*(mi-1) + rep;
            [corefile] = SimMatch_measuredrift_data_func_mem(params);
            f = load([outputfolder,corefile,'.mat']);
            Ds_reps(rep) = nanmean(f.Ds);
            pair_reps(rep) = nanmean(f.pair_drift_vars);
            if isfield(f,'pair_abs_drift_vars')
                pair_abs_reps(rep) = nanmean(f.pair_abs_drift_vars);
            end
            if isfield(f,'pair_drift_incr_vars'); pair_incr_reps(rep) = nanmean(f.pair_drift_incr_vars); end
            if isfield(f,'pair_abs_drift_incr_meanabs'); pair_abs_incr_reps(rep) = nanmean(f.pair_abs_drift_incr_meanabs); end
            if isfield(f,'pred_mse_slope'); pred_mse_slope_reps(rep) = f.pred_mse_slope; end
        end
        Ds_mean(ei) = nanmean(Ds_reps);
        Ds_sem(ei)  = nanstd(Ds_reps)/sqrt(Nreps);
        pair_mean(ei) = nanmean(pair_reps);
        pair_sem(ei)  = nanstd(pair_reps)/sqrt(Nreps);
        pair_abs_mean(ei) = nanmean(pair_abs_reps);
        pair_abs_sem(ei)  = nanstd(pair_abs_reps)/sqrt(Nreps);
        pair_incr_mean(ei) = nanmean(pair_incr_reps);
        pair_incr_sem(ei)  = nanstd(pair_incr_reps)/sqrt(Nreps);
        pair_abs_incr_mean(ei) = nanmean(pair_abs_incr_reps);
        pair_abs_incr_sem(ei)  = nanstd(pair_abs_incr_reps)/sqrt(Nreps);
        pred_mse_slope_mean(ei) = nanmean(pred_mse_slope_reps);
        pred_mse_slope_sem(ei)  = nanstd(pred_mse_slope_reps)/sqrt(Nreps);
    end
    figure('Position',[50,50,1200,350]);
    subplot(1,3,1); errorbar(eig_decays, Ds_mean, Ds_sem, '-o','LineWidth',2); xlabel('eig\\_decay'); ylabel('Mean D'); title(['Drift vs eigen-decay (', mode, ')'],'Interpreter','none');
    subplot(1,3,2); errorbar(eig_decays, pair_mean, pair_sem, '-o','LineWidth',2); xlabel('eig\\_decay'); ylabel('Mean Pair Drift Var'); title(['Pair var vs eigen-decay (', mode, ')'],'Interpreter','none');
    subplot(1,3,3); errorbar(eig_decays, pair_abs_mean, pair_abs_sem, '-o','LineWidth',2); xlabel('eig\\_decay'); ylabel('Mean |Pair| Drift Var'); title(['|Pair| var vs eigen-decay (', mode, ')'],'Interpreter','none');
    saveas(gcf, fullfile(plot_root, ['SM_eigdecay_' mode '_n' num2str(n) '_m' num2str(m) '_rho_' num2str(rho_noise) '.png']));
    figure('Position',[50,50,1200,350]);
    subplot(1,3,1); errorbar(eig_decays, pred_mse_slope_mean, pred_mse_slope_sem, '-o','LineWidth',2); xlabel('eig\\_decay'); ylabel('pred\_mse slope'); title(['MSE drift vs eigen-decay (', mode, ')'],'Interpreter','none');
    subplot(1,3,2); errorbar(eig_decays, pair_incr_mean, pair_incr_sem, '-o','LineWidth',2); xlabel('eig\\_decay'); ylabel('Var(\Delta h_i-\Delta h_j)'); title(['Incr pair var vs eigen-decay (', mode, ')'],'Interpreter','none');
    subplot(1,3,3); errorbar(eig_decays, pair_abs_incr_mean, pair_abs_incr_sem, '-o','LineWidth',2); xlabel('eig\\_decay'); ylabel('E[||\Delta h_i|-|\Delta h_j||]'); title(['Incr |pair| vs eigen-decay (', mode, ')'],'Interpreter','none');
    saveas(gcf, fullfile(plot_root, ['SM_eigdecay_extra_' mode '_n' num2str(n) '_m' num2str(m) '_rho_' num2str(rho_noise) '.png']));
end


