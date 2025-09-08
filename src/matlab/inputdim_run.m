% Run as a function of dimension of input
clear all
close all
m = 5;
outputfolder = 'Results/';
synnoise = 1*1e-2;
Nparticles = 50;
eta = 0.001;
ns = 2:10;
Ds = nan(length(ns),max(ns));
Ds_sem = nan(length(ns),1);
pair_var = nan(length(ns),1);
pair_var_sem = nan(length(ns),1);
pair_abs_incr = nan(length(ns),1);
pair_abs_incr_sem = nan(length(ns),1);
pred_mse_slope = nan(length(ns),1);
pred_mse_slope_sem = nan(length(ns),1);
overwrite = 1;
iter = 0;
Nreps = 3; base_seed = 7777;
sigmaperp = 0.3;
mode = 'excitability'; % or 'synaptic'
rho_noise = 0; % noise correlation coefficient across neurons
% One plot folder for the entire sweep
tstamp = datestr(now,'yyyymmdd_HHMMSS');
plot_root = fullfile('plot', ['inputdim_' mode '_m' num2str(m) '_rho_' num2str(rho_noise) '_' tstamp]);
if ~exist(plot_root,'dir'); mkdir(plot_root); end

for n = ns
    iter = iter + 1;
    Ds_reps = nan(Nreps, min(m,n));
    pair_var_reps = nan(Nreps,1);
    pair_abs_incr_reps = nan(Nreps,1);
    pred_mse_slope_reps = nan(Nreps,1);
    for rep = 1:Nreps
        params = [];
        params.n = n; params.m = m;
        params.Nparticles = Nparticles ; params.synnoise = synnoise;
        params.eta = eta;
        params.numtrialstims = min(m,n);
        params.npool = 8; params.overwrite = overwrite;
        params.outputfolder = outputfolder;
        params.sigmaperp = sigmaperp;
        params.mode = mode;
        params.stdZ = synnoise;
        params.rho_noise = rho_noise;
        % Only generate control plots for first n and first repeat
        params.plotflag = (iter == 1 && rep == 1);
        params.plotfolder = plot_root;
        params.seed = base_seed + 1000*iter + rep;
        [corefile] = SimMatch_measuredrift_data_func_mem(params);
       
        filename = [outputfolder,corefile,'.mat'];
        f = load(filename);
        Ds_reps(rep,1: min(m,n)) = f.Ds(1: min(m,n));
        if isfield(f,'pair_drift_incr_vars'); pair_var_reps(rep) = nanmean(f.pair_drift_incr_vars); end
        if isfield(f,'pair_abs_drift_incr_meanabs'); pair_abs_incr_reps(rep) = nanmean(f.pair_abs_drift_incr_meanabs); end
        if isfield(f,'pred_mse_slope'); pred_mse_slope_reps(rep) = f.pred_mse_slope; end
    end
    Ds(iter,1: min(m,n)) = nanmean(Ds_reps,1);
    Ds_sem(iter) = nanstd(nanmean(Ds_reps,2))/sqrt(Nreps);
    pair_var(iter) = nanmean(pair_var_reps);
    pair_var_sem(iter) = nanstd(pair_var_reps)/sqrt(Nreps);
    pair_abs_incr(iter) = nanmean(pair_abs_incr_reps);
    pair_abs_incr_sem(iter) = nanstd(pair_abs_incr_reps)/sqrt(Nreps);
    pred_mse_slope(iter) = nanmean(pred_mse_slope_reps);
    pred_mse_slope_sem(iter) = nanstd(pred_mse_slope_reps)/sqrt(Nreps);
end
%% Collect results
setupgraphics
figure,
subplot(2,2,1); errorbar(ns,nanmean(Ds(:,:),2),Ds_sem,'-o','LineWidth',2); xline(m,'--','LineWidth',2); xlabel('n'); ylabel('Ds'); title(['SM ', mode], 'Fontsize',12, 'Interpreter','none')
subplot(2,2,2); errorbar(ns, pair_var, pair_var_sem, '-o','LineWidth',2); xline(m,'--','LineWidth',2); xlabel('n'); ylabel('Var(\Delta h_i-\Delta h_j)'); title('Incr pair var','Interpreter','none')
subplot(2,2,3); errorbar(ns, pair_abs_incr, pair_abs_incr_sem, '-o','LineWidth',2); xline(m,'--','LineWidth',2); xlabel('n'); ylabel('E[||\Delta h_i|-|\Delta h_j||]'); title('Incr |pair|','Interpreter','none')
subplot(2,2,4); errorbar(ns, pred_mse_slope, pred_mse_slope_sem, '-o','LineWidth',2); xline(m,'--','LineWidth',2); xlabel('n'); ylabel('pred\_mse slope'); title('MSE drift slope','Interpreter','none')
sgtitle(['Nparticles=',num2str(Nparticles), ', eta=',num2str(eta),', synnoise=',num2str(synnoise), ', m=',num2str(m), ', rho=', num2str(rho_noise)])
if ~exist('plot','dir'); mkdir('plot'); end
saveas(gcf, fullfile(plot_root, ['SM_inputdim_' mode '_m' num2str(m) '_rho_' num2str(rho_noise) '.png']))