import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load recorded data
# The CSV file should contain columns: 'brain_area', 'beta_drift', 'noise_corr', 'noise_corr_change', 'single_neuron_drift', 'pairwise_drift_difference'

data = pd.read_csv('recorded_neural_data.csv')

# Initialize lists to store expected drifts
expected_single_drift_syn = []
expected_pair_drift_syn = []
expected_single_drift_exc = []
expected_pair_drift_exc = []

e = 0.5  # Assume excitability rate (adjust if known)

# Loop through each brain area
for index, row in data.iterrows():
    beta_drift = row['beta_drift']
    noise_corr = row['noise_corr']
    noise_corr_change = row['noise_corr_change']
    
    # Synaptic Noise Model
    expected_single_drift_syn.append(beta_drift)
    expected_pair_drift_syn.append(beta_drift * (1 - noise_corr))
    
    # Excitability Modulation Model
    expected_single_drift_exc.append(2 * e * (1 - e) * row['single_neuron_variance'] ** 0.5)
    expected_pair_drift_exc.append(4 * e * (1 - e) * (row['single_neuron_variance'] - noise_corr * row['pair_covariance']))
    
# Add expected drifts to the data frame
data['expected_single_drift_syn'] = expected_single_drift_syn
data['expected_pair_drift_syn'] = expected_pair_drift_syn
data['expected_single_drift_exc'] = expected_single_drift_exc
data['expected_pair_drift_exc'] = expected_pair_drift_exc

# Save the results
data.to_csv('neural_data_with_expected_drifts.csv', index=False)

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(data['beta_drift'], data['single_neuron_drift'], label='Recorded Single Neuron Drift')
plt.scatter(data['beta_drift'], data['expected_single_drift_syn'], label='Expected Single Drift (Synaptic Noise)')
plt.scatter(data['beta_drift'], data['expected_single_drift_exc'], label='Expected Single Drift (Excitability)')
plt.xlabel('Beta Coefficient Drift')
plt.ylabel('Single Neuron Drift')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data['noise_corr'], data['pairwise_drift_difference'], label='Recorded Pairwise Drift Difference')
plt.scatter(data['noise_corr'], data['expected_pair_drift_syn'], label='Expected Pair Drift Difference (Synaptic Noise)')
plt.scatter(data['noise_corr'], data['expected_pair_drift_exc'], label='Expected Pair Drift Difference (Excitability)')
plt.xlabel('Noise Correlation')
plt.ylabel('Pairwise Drift Difference')
plt.legend()

plt.tight_layout()
plt.show()