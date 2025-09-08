% Grid sweep over rho vs dimensions for both modes; save heatmaps
clear all
close all

outputfolder = 'Results/';
if ~exist(outputfolder,'dir'); mkdir(outputfolder); end
if ~exist('plot','dir'); mkdir('plot'); end
% One plot root for the entire sweep
tstamp = datestr(now,'yyyymmdd_HHMMSS');
plot_root = fullfile('plot', ['grid_' tstamp]);
if ~exist(plot_root,'dir'); mkdir(plot_root); end

synnoise = 1*1e-2;
Nparticles = 24;
eta = 0.001;
sigmaperp = 0.3;
overwrite = 1;
npool = 8;
Nreps = 3;
base_seed = 4242;

ns = 4:10;  % input dims
ms = 4:10;  % output dims
rhos = [-0.2, 0, 0.2, 0.5, 0.8];
modes = {'synaptic','excitability'};

for mi = 1:length(modes)
    mode = modes{mi};
    for ri = 1:length(rhos)
        rho_noise = rhos(ri);
        Ds_mean_grid = nan(length(ns), length(ms));
        pair_var_mean_grid = nan(length(ns), length(ms));
        pair_abs_var_mean_grid = nan(length(ns), length(ms));
        pair_incr_mean_grid = nan(length(ns), length(ms));
        pair_abs_incr_mean_grid = nan(length(ns), length(ms));
        pred_mse_slope_mean_grid = nan(length(ns), length(ms));
        for ii = 1:length(ns)
            for jj = 1:length(ms)
                n = ns(ii); m = ms(jj);
                Ds_reps = nan(Nreps,1);
                pair_reps = nan(Nreps,1);
                pair_abs_reps = nan(Nreps,1);
                for rep = 1:Nreps
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
                    params.plotflag = 0;
                    params.seed = base_seed + rep + 1000*jj + 100000*ii + 10000000*(mi-1) + 19*ri;
                    [corefile] = SimMatch_measuredrift_data_func_mem(params);
                    filename = [outputfolder,corefile,'.mat'];
                    f = load(filename);
                    Ds_reps(rep) = nanmean(f.Ds);
                    pair_reps(rep) = nanmean(f.pair_drift_vars);
                    if isfield(f,'pair_abs_drift_vars')
                        pair_abs_reps(rep) = nanmean(f.pair_abs_drift_vars);
                    end
                end
                if isfield(f,'pair_drift_incr_vars')
                    pair_incr_reps(rep) = nanmean(f.pair_drift_incr_vars);
                else
                    pair_incr_reps(rep) = NaN;
                end
                if isfield(f,'pair_abs_drift_incr_meanabs')
                    pair_abs_incr_reps(rep) = nanmean(f.pair_abs_drift_incr_meanabs);
                else
                    pair_abs_incr_reps(rep) = NaN;
                end
                if isfield(f,'pred_mse_slope')
                    pred_mse_slope_reps(rep) = f.pred_mse_slope;
                else
                    pred_mse_slope_reps(rep) = NaN;
                end
                Ds_mean_grid(ii,jj) = nanmean(Ds_reps);
                pair_var_mean_grid(ii,jj) = nanmean(pair_reps);
                pair_abs_var_mean_grid(ii,jj) = nanmean(pair_abs_reps);
                pair_incr_mean_grid(ii,jj) = nanmean(pair_incr_reps);
                pair_abs_incr_mean_grid(ii,jj) = nanmean(pair_abs_incr_reps);
                pred_mse_slope_mean_grid(ii,jj) = nanmean(pred_mse_slope_reps);
            end
        end
        % Save MAT
        save([outputfolder,'grid_rho_',num2str(rho_noise),'_',mode,'.mat'], ...
            'ns','ms','rho_noise','mode','Ds_mean_grid','pair_var_mean_grid','pair_abs_var_mean_grid','pair_incr_mean_grid','pair_abs_incr_mean_grid','pred_mse_slope_mean_grid');
        % Plot heatmaps
        figure('Position',[50,50,1600,700]);
        subplot(2,3,1); imagesc(ms, ns, Ds_mean_grid); set(gca,'YDir','normal'); colorbar; xlabel('m'); ylabel('n'); title(['Mean D, rho=', num2str(rho_noise),' (',mode,')'],'Interpreter','none');
        subplot(2,3,2); imagesc(ms, ns, pair_var_mean_grid); set(gca,'YDir','normal'); colorbar; xlabel('m'); ylabel('n'); title(['Pair Var, rho=', num2str(rho_noise),' (',mode,')'],'Interpreter','none');
        subplot(2,3,3); imagesc(ms, ns, pair_abs_var_mean_grid); set(gca,'YDir','normal'); colorbar; xlabel('m'); ylabel('n'); title(['|Pair| Var, rho=', num2str(rho_noise),' (',mode,')'],'Interpreter','none');
        subplot(2,3,4); imagesc(ms, ns, pair_incr_mean_grid); set(gca,'YDir','normal'); colorbar; xlabel('m'); ylabel('n'); title(['Incr Pair Var, rho=', num2str(rho_noise),' (',mode,')'],'Interpreter','none');
        subplot(2,3,5); imagesc(ms, ns, pair_abs_incr_mean_grid); set(gca,'YDir','normal'); colorbar; xlabel('m'); ylabel('n'); title(['Incr |Pair|, rho=', num2str(rho_noise),' (',mode,')'],'Interpreter','none');
        subplot(2,3,6); imagesc(ms, ns, pred_mse_slope_mean_grid); set(gca,'YDir','normal'); colorbar; xlabel('m'); ylabel('n'); title(['MSE slope, rho=', num2str(rho_noise),' (',mode,')'],'Interpreter','none');
        saveas(gcf, fullfile(plot_root, ['SM_grid_rho_' num2str(rho_noise) '_' mode '.png']));
    end
end


