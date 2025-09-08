# Drift/FromFarhad – Simulation and Analysis Guide

This folder contains MATLAB code to simulate representational drift in a similarity-matching network under two mechanisms:
- synaptic: noisy Hebbian/anti-Hebbian updates to `W` and `M` with controllable across-neuron noise correlations.
- excitability: noisy per-neuron excitability factor `zeta` that modulates feedforward drive, with an option for correlated or independent excitability noise.

## Key files

- `SimMatch_measuredrift_data_func_mem_hs.m`
  - Core simulator and measurement pipeline.
  - Inputs via `params` (see Parameters below).
  - Pretrains to a stable PSP solution, then simulates drift.
  - Measures:
    - `Ds`: diffusion/slopes from autocorrelation decay of responses.
    - `pair_drift_vars`: variance of signed pairwise drift differences.
    - `pair_abs_drift_vars`: variance of absolute-magnitude pairwise drift differences.
    - `pair_noise_corrs` and `pair_signal_corrs` from the pretraining window.
  - Saves results into `Results/ResultSimMatch_*.mat` and diagnostic plots into `plot/`.

- `noisecorr_run_hs.m`
  - Sweeps across noise correlation `rho` for both mechanisms.
  - Aggregates repeats and plots:
    - Mean `D` vs `rho`.
    - Mean pair drift variance vs `rho`.
    - Mean |pair| drift variance vs `rho`.
    - Correlation of pair drift variance with noise and signal correlations vs `rho`.
  - Saves summary to `plot/SM_rho_sweep_*.png` (plus `_corrs_*.png`).

- `dim_rho_grid_hs.m`
  - Grid sweep across input dimension `n`, output dimension `m`, and noise correlation `rho`, for both mechanisms.
  - Produces heatmaps of mean `D`, pair drift variance, and |pair| drift variance.
  - Saves figures as `plot/SM_grid_rho_<rho>_<mode>.png` and summaries in `Results/`.

- `eigdecay_run_hs.m`
  - Sweeps the input covariance eigenvalue decay exponent (`eig_decay`) to probe input complexity effects.
  - Runs both mechanisms and plots `D`, pair drift variance, and |pair| drift variance vs `eig_decay`.

- `inputdim_run_hs.m` / `outputdim_run_hs.m`
  - 1D sweeps for `n` (fixed `m`) or `m` (fixed `n`) used for quick `D` vs dimension visuals; prefer the grid script for comprehensive coverage.

- `run_vs_dim.m`
  - Legacy combined dimension sweeps; prefer `dim_rho_grid_hs.m` for comprehensive analysis.

- `setupgraphics.m`
  - Plot styling helper.

- `run_matlab_job_rho_hs.sh`, `run_matlab_job.sh`
  - Example SLURM job scripts for batch runs on the cluster.

## Functions and scripts for analysis (what each shows/quantifies)

- `SimMatch_measuredrift_data_func_mem_hs.m` (core)
  - Runs pretraining to reach PSP; then drift simulation under synaptic/excitability mechanisms.
  - Quantifies:
    - Ds: slope from −log Rauto(t) (representational drift rate).
    - Pair metrics: Var(h_i − h_j), Var(|h_i| − |h_j|).
    - Increment metrics: Var(Δh_i − Δh_j), E[||Δh_i| − |Δh_j||].
    - GLM generalization drift: pred_mse(D) using β(t−D) on x(t), plus slope pred_mse_slope.
    - Correlations: pair_noise_corrs, pair_signal_corrs from fixed-input windows.
  - Visualizations (when plotflag=1): pretraining loss; drift_vs_correlations scatters; pred_mse vs D curves; Rauto(t) curves; PSP projector similarity; per-neuron MSE slope histogram.

- `noisecorr_run_hs.m` (ρ sweeps)
  - Sweeps noise correlation ρ for each mechanism; repeats across seeds.
  - Plots vs ρ: Ds; Var(h_i − h_j); Var(|h_i| − |h_j|); Var(Δh_i − Δh_j); E[||Δh_i| − |Δh_j||]; pred_mse_slope.
  - Correlation summaries: corr(pair drift var, noise/signal corr) with error bars.

- `dim_rho_grid_hs.m` (n×m×ρ grids)
  - Heatmaps across (n, m) for a given ρ and mechanism.
  - Shows: Ds; pair var; |pair| var; increment metrics; pred_mse_slope.

- `eigdecay_run_hs.m` (input complexity)
  - Sweeps eigenvalue power-law exponent eig_decay.
  - Shows: Ds; pair var; |pair| var; increment metrics; pred_mse_slope vs eig_decay.

- `inputdim_run_hs.m`
  - Varies input dimension n at fixed m.
  - Shows: Ds; Var(Δh_i − Δh_j); E[||Δh_i| − |Δh_j||]; pred_mse_slope vs n.

- `outputdim_run_hs.m`
  - Varies output dimension m at fixed n (simple 1D view; for full (n,m) use the grid script).
  - Shows Ds vs n label; extend similarly if needed.

- `run_vs_dim.m` (legacy)
  - Older combined dimension sweep without the newer metrics/plots.
  - Prefer the `_hs` scripts above for current analyses; this file is kept for reference.

- `SimMatch_measuredrift_data_func_mem.m` (older core)
  - Earlier version lacking correlated-noise controls and extended metrics; kept for reference.

- `setupgraphics.m`
  - Sets plotting defaults.

## Parameters (common `params` fields)

- `n`, `m`: input and output dimensions.
- `mode`: `'synaptic'` or `'excitability'`.
- `synnoise`: scalar standard deviation of the stochastic increments per time step (per √dt) used for synaptic noise (in W, M) or excitability noise (`stdZ` defaults to this). Larger values increase drift.
- `rho_noise`: across-neuron noise correlation (used to build a valid correlation matrix for noise injection).
- `excit_corr`: for `mode='excitability'`, set `1` for correlated excitability noise (matches `rho_noise`), or `0` for independent excitability noise (matches analytical assumption of independence).
- `eta`: learning rate (also used as the simulation time step dt). Smaller `eta` slows learning/drift updates; larger `eta` increases update magnitude.
- `Tpretrain`: number of pretraining update steps to reach/stabilize the principal subspace solution.
- `T`: number of drift simulation steps per particle used to measure autocorrelation decay and drift statistics.
- `numtrialstims`: number of orthonormal trial stimuli (from top singular vectors of input) used to probe responses; defaults to `min(m,n)`.
- `sigmapar`, `sigmaperp`, or `sigmas`: parameters controlling the input covariance eigenvalues. If `eig_decay` is unset, eigenvalues are constructed as `[sigmapar*ones(min(m,n),1); sigmaperp*ones(abs(n-m),1)]` (reordered by random orthogonal `Q`). Alternatively, provide explicit `sigmas`.
- `eig_decay`: optional power-law exponent for input eigenvalue decay (λ_i ∝ i^{-eig_decay}); if set, overrides `sigmas`/`sigmapar`.
- `Nparticles`: number of repeated drift simulations (particles) per configuration used to average drift metrics and reduce variance.
- `npool`: number of particles simulated in a batch (inner loop) before aggregating; controls memory/compute chunking, not MATLAB Parallel Toolbox.
- `seed`: RNG seed for reproducibility (sets `rng(seed,'twister')`).
- `overwrite`: if 0 and result file exists, skip recomputing; if 1, recompute and overwrite results.
- `plotflag`: enable/disable per-run diagnostic plotting.
- `outputfolder`: where to write `ResultSimMatch_*.mat`.
- `excit_corr`: for `mode='excitability'`, set 1 for correlated excitability noise across neurons, or 0 for independent increments.
- `excit_rho`: pairwise correlation coefficient for excitability increments when `excit_corr=1`. If empty, it defaults to match `rho_noise`.

## How to run

### 1) Noise correlation sweeps (rho)
- Interactive MATLAB:
```matlab
noisecorr_run_hs
```
- SLURM:
```bash
sbatch run_matlab_job_rho_hs.sh
```
- Outputs:
  - `plot/SM_rho_sweep_<mode>_n8_m8.png`
  - `plot/SM_rho_sweep_corrs_<mode>_n8_m8.png`
  - `plot/SM_rho_sweep_extra_<mode>_n8_m8.png`
  - `Results/rho_sweep_summary.mat`
- To match analytical “independent excitability” assumptions, edit `noisecorr_run_hs.m` to set `params.excit_corr = 0` when `mode='excitability'`.

### 2) Dimension × rho grids
- Interactive MATLAB:
```matlab
dim_rho_grid_hs
```
- SLURM:
```bash
sbatch --wrap "module load matlab/R2024b-fasrc01; matlab -nodisplay -nosplash -nodesktop -r 'dim_rho_grid_hs; exit'"
```
- Outputs:
  - `plot/SM_grid_rho_<rho>_<mode>.png`
  - `Results/grid_rho_<rho>_<mode>.mat`

### 3) Input complexity (eigen-decay) sweeps
- Interactive MATLAB:
```matlab
eigdecay_run_hs
```
- SLURM:
```bash
sbatch --wrap "module load matlab/R2024b-fasrc01; matlab -nodisplay -nosplash -nodesktop -r 'eigdecay_run_hs; exit'"
```
- Outputs:
  - `plot/SM_eigdecay_<mode>_n12_m12_rho_0.2.png`
  - `plot/SM_eigdecay_extra_<mode>_n12_m12_rho_0.2.png`
  - Individual `ResultSimMatch_*.mat` files in `Results/`

### 4) Simple dimension-only sweeps
- For quick `D` vs `n` or `m` curves:
```matlab
inputdim_run_hs    % varies n at fixed m
outputdim_run_hs   % varies m at fixed n
```

## Notes on interpretation (consistency with model)
- Synaptic model:
  - Pair drift difference variance decreases with increasing `rho_noise` (shared noise cancels in differences).
  - `D` scales with effective subspace dimension `k=min(n,m)` and depends on input eigen-spectrum; stronger concentration of variance (larger `eig_decay`) lowers `D`.
- Excitability model:
  - With independent excitability (`excit_corr=0`), pair drift difference variance is flat vs `rho_noise` and scales with neuron count; set this flag to match analytical assumptions.
  - With correlated excitability (`excit_corr=1`), pair drift difference variance can also decrease with `rho_noise` due to shared fluctuations.

## Outputs
- `Results/ResultSimMatch_*.mat`: per-run results with fields saved by the core function.
- `Results/*summary*.mat`: sweep summary structures.
- `plot/*.png`: figures for all sweeps and pretraining diagnostics.

## Visualizations and diagnostics

### PSP verification and representational similarity stability
- During pretraining, the simulator logs a “subspace loss” comparing the learned operator `F = M^{-1}W` (or excitability equivalent) to the top-`k` principal subspace. See:
  - `plot/pretraining_loss_<mode>_n<n>_m<m>_rho_<rho>.png`
- After pretraining, drift is measured on fixed trial stimuli `trialstims` drawn from the learned subspace. The autocorrelation of responses `Rauto(t)` is computed and `-log Rauto` is fit to obtain diffusion-like `D`. Stable representational similarity manifests as a slow decay curve; `D` summarizes this decay.

### Drift metrics (what is computed and plotted)
- Ds: slope from linear regression of `-log Rauto(t)` vs time.
- Pairwise drift (signed): `Var(h_i - h_j)` over time (per stimulus, then averaged).
- Magnitude difference: `Var(|h_i| - |h_j|)`.
- Increment-based (drift differences using per-step changes, aligning with Δβ):
  - `Var(Δh_i - Δh_j)`.
  - `E[||Δh_i| - |Δh_j||]` (use this to test “pairs with similar drift magnitudes have higher noise correlation”).
- GLM prediction-error drift (new):
  - For lags `D_eval`, compute MSE of predicting `y(t)` with the old operator `B(t-D)` on current `x(t)`: `||y(t) - B(t-D)x(t)||^2`.
  - Report `pred_mse(D)` and a single slope `pred_mse_slope` from a linear fit.
- Where to see them:
  - `plot/SM_rho_sweep_<mode>_*.png`: Ds, pair var, |pair| var vs ρ.
  - `plot/SM_rho_sweep_extra_<mode>_*.png`: `pred_mse_slope`, `Var(Δh_i-Δh_j)`, `E[||Δh_i|-|Δh_j||]` vs ρ.
  - `plot/SM_grid_rho_<rho>_<mode>.png`: heatmaps for Ds, pair var, |pair| var, increment metrics, and `pred_mse_slope` across (n, m).
  - `plot/SM_eigdecay_*_*.png` and `*_extra_*.png`: metrics vs `eig_decay`.
  - `plot/drift_vs_correlations_<mode>_n<n>_m<m>_rho_<rho>.png`: scatter plots of pair metrics vs noise/signal correlations.
  - `plot/mse_drift_<mode>_n<n>_m<m>_rho_<rho>.png`: `pred_mse(D)` curves with a fitted line and an “MSE increase from D=0” view.

### How drift is computed (t=0 to T)
- Fix trial stimuli `trialstims` in the learned subspace.
- At each time step `t ∈ [0, T]`, compute `h(t, s) = M(t)^{-1}W(t)s` (or with excitability `M^{-1}diag(ζ(t))W s`) for each stimulus `s`.
- Compute `Rauto(t)` using `h(0, s)` and `h(t, s)`. Fit a line to `-log Rauto(t)` to get `D` per `s`; average across `s`.
- Pairwise drift metrics use the time series `h_i(t, s)`; increment metrics use `Δh_i(t, s) = h_i(t+1, s) - h_i(t, s)`.

### Time windows and sensitivity
- Pretraining length `Tpretrain`:
  - Longer pretraining reduces subspace error and stabilizes initial `h(0, s)`. Use `1e5` by default; reduce for speed at the cost of slightly larger `D` and noisier noise-correlation estimates.
- Drift length `T` and `Nparticles`:
  - Longer `T` (and/or larger `Nparticles`) reduces variance of `D`, pair, and increment metrics, and improves `pred_mse(D)` fits. Defaults: `T=1e4`, `Nparticles` 24–50.
  - If `T` is small, `D` fits may be noisy and `pred_mse_slope` less reliable.
- `D_eval` (lags for prediction-error drift):
  - Defaults to `[1 2 5 10 20 50 100]`. Increase to probe longer generalization horizons; slope may become more linear/robust with more points.
- Noise correlation range (`rho_noise`) and excitability correlation (`excit_rho`):
  - `rho_noise` sweeps the shared-noise coordination (synaptic). `excit_rho` controls excitability correlation when `excit_corr=1`.

### Suggested visual sanity checks
- PSP stability: inspect `pretraining_loss_*` and verify the loss plateaus low; optionally compute and plot `F(t)'F(t)` similarity to top-`k` projector.
- Correlation structure: check `drift_vs_correlations_*` scatter plots to verify the expected negative relation between pair drift difference and noise correlation (for similarly tuned pairs) and reduced magnitude-difference with larger |ρ|.
- Drift increase with time lag: check `mse_drift_*` curves; high-drift neurons show faster MSE growth vs lag.

## FAQ and guidance
- How to change noise correlation independently for excitability?
  - Set `excit_corr=1` and choose `excit_rho` (e.g., 0.5). If `excit_rho` is empty, it matches `rho_noise`.
- How to enforce independent excitability increments (to match the analytical model)?
  - Set `excit_corr=0` (e.g., in `noisecorr_run_hs.m` via `excit_corr_excitability = 0`).
- How to choose `Tpretrain`, `T`, and `Nparticles`?
  - Increase `Tpretrain` until the pretraining loss plot stabilizes. Increase `T`/`Nparticles` until error bars on your key plots are acceptable.
