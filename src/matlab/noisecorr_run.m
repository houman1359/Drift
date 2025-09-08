% Sweep noise correlation for both synaptic and excitability modes
% PARAMETERS (edit here)
%   n, m                 : input/output dims
%   synnoise             : base noise std (used as stdW/stdM or stdZ)
%   Nparticles           : # particles per chunk
%   eta                  : learning rate / dt
%   sigmaperp            : orthogonal variance scale for inputs
%   overwrite            : skip if results exist (0/1)
%   npool                : particles concurrently simulated
%   excit_corr_excitability : 0 => indep excit noise; 1 => correlated
%   excit_rho_excitability  : equicorr coeff for excitability (optional)
%   rhos                 : synaptic equicorr coefficients to sweep
%   modes                : {'synaptic','excitability'}
%   Nreps, base_seed     : repeats and seed base for error bars
% Notes:
%   - Empirical residual noise corr is computed inside the core function.
%   - For heterogeneous/custom correlations, set params.noise_corr_mode /
%     params.excit_corr_mode and related fields below.
clear all
close all

outputfolder = 'Results/';
if ~exist(outputfolder,'dir'); mkdir(outputfolder); end
if ~exist('plot','dir'); mkdir('plot'); end

% Base params
n = 6; m = 6;
rng('default');
% single plot root for entire sweep (after n,m defined)
tstamp = datestr(now,'yyyymmdd_HHMMSS');
plot_root = fullfile('plot', ['rho_sweep_' num2str(n) 'x' num2str(m) '_' tstamp]);
if ~exist(plot_root,'dir'); mkdir(plot_root); end
synnoise = 1*1e-2;
Nparticles = 32;
eta = 0.001;
sigmaperp = 0.3;
overwrite = 1;
npool = 8;

rng('default');
% Toggle: excitability noise correlation across neurons.
% 0 => independent excit noise; 1 => correlated (structure set below)
excit_corr_excitability = 1;
% Optional: excitability equicorrelation coefficient (used if excit_corr_mode='equicorr').
% If empty, it will match rho_noise inside the core function.
excit_rho_excitability = [];

% Correlation structure controls (defaults: synaptic equicorr; excitability heterogeneous)
noise_corr_mode = 'equicorr';   % 'equicorr' | 'matrix' | 'hetero'
alpha_range     = [0.0 0.0];    % for synaptic 'hetero'
excit_corr_mode = 'hetero';     % 'match' | 'equicorr' | 'matrix' | 'hetero'
alpha_ex_range  = [0.0 0.6];    % for excitability 'hetero'

rhos = [-0.2  , -0.1, 0, 0.1, 0.2];
modes = {'excitability'}; % set to {'synaptic','excitability'} to run both
Nreps = 20; % repeats for error bars
base_seed = 12345;

results = struct();
for mi = 1:length(modes)
    mode = modes{mi};
    Ds_mean = nan(length(rhos),1);
    Ds_sem = nan(length(rhos),1);
    pair_var_mean = nan(length(rhos),1);
    pair_var_sem = nan(length(rhos),1);
    pair_abs_var_mean = nan(length(rhos),1);
    pair_abs_var_sem = nan(length(rhos),1);
    r_noise_emp_mean = nan(length(rhos),1);
    r_noise_emp_sem = nan(length(rhos),1);
    r_signal_mean = nan(length(rhos),1);
    r_signal_sem = nan(length(rhos),1);
    % New drift metrics
    pair_incr_var_mean = nan(length(rhos),1);
    pair_incr_var_sem = nan(length(rhos),1);
    pair_abs_incr_meanabs_mean = nan(length(rhos),1);
    pair_abs_incr_meanabs_sem = nan(length(rhos),1);
    pred_mse_slope_mean = nan(length(rhos),1);
    pred_mse_slope_sem = nan(length(rhos),1);
    for ri = 1:length(rhos)
        rho_noise = rhos(ri);
        Ds_reps = nan(Nreps,1);
        pair_var_reps = nan(Nreps,1);
        pair_abs_var_reps = nan(Nreps,1);
        pair_incr_var_reps = nan(Nreps,1);
        pair_abs_incr_meanabs_reps = nan(Nreps,1);
        pred_mse_slope_reps = nan(Nreps,1);
        for rep = 1:Nreps
            [mi ri rep]
            params = [];
            params.n = n; params.m = m;
            params.Nparticles = Nparticles ; params.synnoise = synnoise;
            params.eta = eta;
            params.numtrialstims = min(m,n);
            params.npool = npool; params.overwrite = overwrite;
            params.outputfolder = outputfolder;
            params.sigmaperp = sigmaperp;
            params.mode = mode;
            params.stdZ = synnoise;
            params.rho_noise = rho_noise;
            % Correlation structure toggles
            params.noise_corr_mode = noise_corr_mode;
            params.alpha_range = alpha_range;
            if strcmp(mode,'excitability')
                params.excit_corr = excit_corr_excitability;
                params.excit_rho = excit_rho_excitability;
                params.excit_corr_mode = excit_corr_mode;
                params.alpha_ex_range = alpha_ex_range;
            end
            % Only generate per-run control plots for one representative config
            make_plots = (mi == 1) && (ri == ceil(length(rhos)/2)) && (rep == 1);
            params.plotflag = make_plots;
            params.plotfolder = plot_root;
            params.seed = base_seed + rep + 1000*ri + 100000*(mi-1);
            [corefile] = SimMatch_measuredrift_data_func_mem(params);
            filename = [outputfolder,corefile,'.mat'];
            f = load(filename);
            Ds_reps(rep) = nanmean(f.Ds);
            pair_var_reps(rep) = nanmean(f.pair_drift_vars);
            if isfield(f,'pair_abs_drift_vars')
                pair_abs_var_reps(rep) = nanmean(f.pair_abs_drift_vars);
            else
                pair_abs_var_reps(rep) = NaN;
            end
            if isfield(f,'pair_drift_incr_vars')
                pair_incr_var_reps(rep) = nanmean(f.pair_drift_incr_vars);
            end
            if isfield(f,'pair_abs_drift_incr_meanabs')
                pair_abs_incr_meanabs_reps(rep) = nanmean(f.pair_abs_drift_incr_meanabs);
            end
            if isfield(f,'pred_mse_slope')
                pred_mse_slope_reps(rep) = f.pred_mse_slope;
            end
            rn_emp = corr(f.pair_noise_corrs_emp, f.pair_drift_vars, 'Rows','pairwise');
            rs = corr(f.pair_signal_corrs, f.pair_drift_vars, 'Rows','pairwise');
            if ~exist('r_noise_emp_reps','var') || numel(r_noise_emp_reps) < Nreps
                r_noise_emp_reps = nan(Nreps,1);
                r_signal_reps = nan(Nreps,1);
            end
            r_noise_emp_reps(rep) = rn_emp;
            r_signal_reps(rep) = rs;
        end
        Ds_mean(ri) = nanmean(Ds_reps);
        Ds_sem(ri) = nanstd(Ds_reps)/sqrt(Nreps);
        pair_var_mean(ri) = nanmean(pair_var_reps);
        pair_var_sem(ri) = nanstd(pair_var_reps)/sqrt(Nreps);
        pair_abs_var_mean(ri) = nanmean(pair_abs_var_reps);
        pair_abs_var_sem(ri) = nanstd(pair_abs_var_reps)/sqrt(Nreps);
        r_noise_emp_mean(ri) = nanmean(r_noise_emp_reps);
        r_noise_emp_sem(ri) = nanstd(r_noise_emp_reps)/sqrt(Nreps);
        r_signal_mean(ri) = nanmean(r_signal_reps);
        r_signal_sem(ri) = nanstd(r_signal_reps)/sqrt(Nreps);
        pair_incr_var_mean(ri) = nanmean(pair_incr_var_reps);
        pair_incr_var_sem(ri) = nanstd(pair_incr_var_reps)/sqrt(Nreps);
        pair_abs_incr_meanabs_mean(ri) = nanmean(pair_abs_incr_meanabs_reps);
        pair_abs_incr_meanabs_sem(ri) = nanstd(pair_abs_incr_meanabs_reps)/sqrt(Nreps);
        pred_mse_slope_mean(ri) = nanmean(pred_mse_slope_reps);
        pred_mse_slope_sem(ri) = nanstd(pred_mse_slope_reps)/sqrt(Nreps);
    end
    results.(mode).rhos = rhos;
    results.(mode).Ds_mean = Ds_mean;
    results.(mode).Ds_sem = Ds_sem;
    results.(mode).pair_var_mean = pair_var_mean;
    results.(mode).pair_var_sem = pair_var_sem;
    results.(mode).pair_abs_var_mean = pair_abs_var_mean;
    results.(mode).pair_abs_var_sem = pair_abs_var_sem;
    results.(mode).r_noise_emp_mean = r_noise_emp_mean;
    results.(mode).r_noise_emp_sem = r_noise_emp_sem;
    results.(mode).r_signal_mean = r_signal_mean;
    results.(mode).r_signal_sem = r_signal_sem;
    results.(mode).pair_incr_var_mean = pair_incr_var_mean;
    results.(mode).pair_incr_var_sem = pair_incr_var_sem;
    results.(mode).pair_abs_incr_meanabs_mean = pair_abs_incr_meanabs_mean;
    results.(mode).pair_abs_incr_meanabs_sem = pair_abs_incr_meanabs_sem;
    results.(mode).pred_mse_slope_mean = pred_mse_slope_mean;
    results.(mode).pred_mse_slope_sem = pred_mse_slope_sem;
    % Plot summary
    figure;
    subplot(1,3,1); errorbar(rhos, Ds_mean, Ds_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('Mean D'); title(['Drift vs \rho (', mode, ')'],'Interpreter','none');
    subplot(1,3,2); errorbar(rhos, pair_var_mean, pair_var_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('Mean Pair Drift Var'); title(['Pair var vs \rho (', mode, ')'],'Interpreter','none');
    subplot(1,3,3); errorbar(rhos, pair_abs_var_mean, pair_abs_var_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('Mean |Pair| Drift Var'); title(['|Pair| var vs \rho (', mode, ')'],'Interpreter','none');
    saveas(gcf, fullfile(plot_root, ['SM_rho_sweep_' mode '_n' num2str(n) '_m' num2str(m) '.png']));
    % New metrics summary
    figure;
    subplot(1,3,1); errorbar(rhos, pred_mse_slope_mean, pred_mse_slope_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('pred\_mse slope'); title(['MSE drift vs \rho (', mode, ')'],'Interpreter','none');
    subplot(1,3,2); errorbar(rhos, pair_incr_var_mean, pair_incr_var_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('Var(\Delta h_i-\Delta h_j)'); title(['Incr pair var vs \rho (', mode, ')'],'Interpreter','none');
    subplot(1,3,3); errorbar(rhos, pair_abs_incr_meanabs_mean, pair_abs_incr_meanabs_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('E[||\Delta h_i|-|\Delta h_j||]'); title(['Incr |pair| vs \rho (', mode, ')'],'Interpreter','none');
    saveas(gcf, fullfile(plot_root, ['SM_rho_sweep_extra_' mode '_n' num2str(n) '_m' num2str(m) '.png']));
    figure;
    subplot(1,2,1); errorbar(rhos, r_noise_emp_mean, r_noise_emp_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('corr(emp noise corr, pair drift var)'); title(['Pair drift vs empirical noise corr (',mode,')'],'Interpreter','none');
    subplot(1,2,2); errorbar(rhos, r_signal_mean, r_signal_sem, '-o','LineWidth',2); xlabel('\rho'); ylabel('corr(pair signal corr, pair drift var)'); title(['Pair drift vs signal corr (',mode,')'],'Interpreter','none');
    saveas(gcf, fullfile(plot_root, ['SM_rho_sweep_corrs_' mode '_n' num2str(n) '_m' num2str(m) '.png']));
end
save([outputfolder, 'rho_sweep_summary.mat'],'results');

