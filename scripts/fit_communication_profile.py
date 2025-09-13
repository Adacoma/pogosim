#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model : Simple Power Law
P_success = 1/(1 + a * msg_size^b * p_send^c * cluster_size^d)
"""

import json
import numpy as np
import pandas as pd
from io import StringIO
import cma
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

sns.set(font_scale=1.0)

# Communication profile experimental data
DATA_CSV = r"""
payload_size,neighbors_per_probe,p_send,success_probability,std_deviation
3,6,0.1,0.996,0.0
3,6,0.2,0.976,0.0
3,6,0.5,0.833,0.0
3,6,0.8,0.604,0.0
3,6,1.0,0.429,0.0
3,2,0.1,0.972,0.0
3,2,0.2,0.978,0.0
3,2,0.5,0.988,0.0
3,2,0.8,0.997,0.0
3,2,1.0,0.996,0.0
3,1,0.1,0.968,0.0
3,1,0.2,0.980,0.0
3,1,0.5,0.982,0.0
3,1,0.8,0.996,0.0
3,1,1.0,1.000,0.0
20,6,0.1,0.995,0.0
20,6,0.2,0.988,0.0
20,6,0.5,0.903,0.0
20,6,0.8,0.349,0.0
20,6,1.0,0.272,0.0
20,2,0.1,0.997,0.0
20,2,0.2,0.998,0.0
20,2,0.5,0.998,0.0
20,2,0.8,0.997,0.0
20,2,1.0,0.999,0.0
20,1,0.1,0.997,0.0
20,1,0.2,0.996,0.0
20,1,0.5,1.000,0.0
20,1,0.8,0.988,0.0
20,1,1.0,1.000,0.0
100,6,0.1,0.772,0.0
100,6,0.2,0.366,0.0
100,6,0.5,0.023,0.0
100,6,0.8,0.013,0.0
100,6,1.0,0.006,0.0
100,2,0.1,0.725,0.0
100,2,0.2,0.779,0.0
100,2,0.5,0.645,0.0
100,2,0.8,0.351,0.0
100,2,1.0,0.080,0.0
100,1,0.1,0.707,0.0
100,1,0.2,0.753,0.0
100,1,0.5,0.637,0.0
100,1,0.8,0.336,0.0
100,1,1.0,0.047,0.0
"""

def load_data():
    """Load and preprocess the experimental data"""
    rows = []
    for line in DATA_CSV.strip().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(s)
    
    df = pd.read_csv(StringIO("\n".join(rows)))
    df["cluster_size"] = df["neighbors_per_probe"] + 1  # Add the node itself
    df["msg_size"] = df["payload_size"] + 3  # Add protocol overhead
    return df

def sigmoid(x):
    """Sigmoid activation function for parameter mapping"""
    return 1.0 / (1.0 + np.exp(-x))

def model_simple(params, msg_size, cluster_size, p_send):
    """
    Simple communication success model
    P_success = 1/(1 + a * msg_size^b * p_send^c * cluster_size^d)
    
    Args:
        params: Dictionary containing model parameters {a, b, c, d}
        msg_size: Message size values
        cluster_size: Cluster size values
        p_send: Sending probability values
    
    Returns:
        Predicted success probabilities
    """
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    
    # Calculate the term inside the denominator
    term = a * (msg_size ** b) * (p_send ** c) * (cluster_size ** d)
    
    return 1.0 / (1.0 + term)

def map_parameters_model(u):
    """
    Map unconstrained optimization variables to model parameters
    
    Args:
        u: Array of 4 unconstrained variables
    
    Returns:
        Dict of mapped parameters with appropriate ranges
    """
    assert len(u) == 4, f"Expected 4 parameters, got {len(u)}"
    
    # Parameter mapping with reasonable ranges
    a = np.exp(-15 + 20 * sigmoid(u[0]))   # Scale factor [exp(-15), exp(5)] ≈ [3e-7, 148]
    b = 0.1 + 4.9 * sigmoid(u[1])          # msg_size power [0.1, 5.0]
    c = 0.1 + 4.9 * sigmoid(u[2])          # p_send power [0.1, 5.0]
    d = -2.0 + 7.0 * sigmoid(u[3])         # cluster_size power [-2.0, 5.0]
    
    return {
        'a': a,
        'b': b,
        'c': c,
        'd': d
    }

def huber_loss(residuals, delta=1.35):
    """
    Huber loss - less sensitive to outliers than squared loss
    Combines quadratic loss for small residuals with linear loss for large residuals
    
    Args:
        residuals: Array of residuals
        delta: Threshold for switching from quadratic to linear
    
    Returns:
        Mean Huber loss
    """
    abs_residuals = np.abs(residuals)
    quadratic = abs_residuals <= delta
    linear = abs_residuals > delta
    
    loss = np.zeros_like(residuals)
    loss[quadratic] = 0.5 * residuals[quadratic] ** 2
    loss[linear] = delta * (abs_residuals[linear] - 0.5 * delta)
    
    return np.mean(loss)

def compute_comprehensive_metrics(y_true, y_pred):
    """Compute comprehensive performance metrics"""
    residuals = y_true - y_pred
    
    # Basic metrics
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    
    # Additional statistical measures
    mape = float(np.mean(np.abs(residuals / y_true)) * 100)  # Mean Absolute Percentage Error
    kurtosis = float(stats.kurtosis(residuals))  # Excess kurtosis (0 for normal)
    skewness = float(stats.skew(residuals))      # Asymmetry measure
    
    # Robust metrics
    huber_loss_val = float(huber_loss(residuals))
    median_abs_error = float(np.median(np.abs(residuals)))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'huber_loss': huber_loss_val,
        'median_abs_error': median_abs_error,
        'residuals': residuals
    }

def fit_model_robust(max_iter=8000, seed=42):
    """
    Fit the communication model using CMA-ES optimization with larger budget
    
    Args:
        max_iter: Maximum iterations for optimization (increased from 3000)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (parameters, metrics, dataframe_with_predictions, convergence_history)
    """
    print("Loading experimental data...")
    df = load_data()
    
    # Extract variables
    y = df["success_probability"].values
    msg_size = df["msg_size"].values
    cluster_size = df["cluster_size"].values
    p_send = df["p_send"].values
    
    print(f"Loaded {len(y)} data points")
    print(f"msg_size range: [{np.min(msg_size)}, {np.max(msg_size)}]")
    print(f"cluster_size range: [{np.min(cluster_size)}, {np.max(cluster_size)}]")
    print(f"p_send range: [{np.min(p_send)}, {np.max(p_send)}]")
    print(f"success_probability range: [{np.min(y):.6f}, {np.max(y):.6f}]")
    
    convergence_history = []
    
    def objective_function(u):
        """Objective function for optimization (minimize Huber loss)"""
        try:
            params = map_parameters_model(u)
            y_pred = model_simple(params, msg_size, cluster_size, p_send)
            
            # Sanity checks for predictions
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return 1000.0
            if np.any(y_pred < 0) or np.any(y_pred > 2.0):
                return 1000.0
            
            # Use Huber loss (more robust to outliers)
            residuals = y - y_pred
            return huber_loss(residuals)
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000.0
    
    # Initialize optimization with larger population and budget
    initial_guess = np.array([0.0, 1.5, 2.0, 1.0])  # 4 parameters
    
    print(f"\nStarting CMA-ES optimization with {max_iter} max iterations...")
    print(f"Model: P_success = 1/(1 + a × msg_size^b × p_send^c × cluster_size^d)")
    print(f"Initial guess: {initial_guess}")
    
    # CMA-ES settings with larger budget
    es = cma.CMAEvolutionStrategy(
        initial_guess, 
        0.3,  # Initial step size
        {
            'seed': seed,
            'maxiter': max_iter,        # Increased from 3000
            'popsize': 50,              # Increased population size
            'verb_disp': 200,           # Display every 200 iterations
            'verb_log': 0,              # No logging
            'tolfun': 1e-12,            # Tighter function tolerance
            'tolx': 1e-12,              # Tighter parameter tolerance
            'timeout': 1800             # 30 minute timeout
        }
    )
    
    iteration = 0
    best_loss = float('inf')
    
    while not es.stop():
        X = es.ask()
        fitness_values = [objective_function(x) for x in X]
        es.tell(X, fitness_values)
        
        iteration += 1
        current_best = min(fitness_values)
        convergence_history.append(current_best)
        
        if current_best < best_loss:
            best_loss = current_best
            
        if iteration % 200 == 0:
            print(f"Iteration {iteration}: Best Huber Loss = {current_best:.8f}")
    
    # Get final results
    u_optimal = es.best.x
    params_optimal = map_parameters_model(u_optimal)
    y_pred_optimal = model_simple(params_optimal, msg_size, cluster_size, p_send)
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(y, y_pred_optimal)
    
    # Create results dataframe
    df_results = df.copy()
    df_results['predicted'] = y_pred_optimal
    df_results['absolute_error'] = np.abs(y - y_pred_optimal)
    df_results['relative_error'] = np.abs(y - y_pred_optimal) / y * 100
    df_results['residuals'] = metrics['residuals']
    
    print(f"\nOptimization completed after {iteration} iterations")
    print(f"Final Huber Loss: {best_loss:.8f}")
    
    return params_optimal, metrics, df_results, convergence_history

def print_detailed_results(params, metrics, df_results):
    """Print comprehensive results analysis"""
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(f"Model: P_success = 1/(1 + a × msg_size^b × p_send^c × cluster_size^d)")
    print(f"Parameters: 4 (a, b, c, d)")
    print()
    
    # Model parameters
    print("FITTED PARAMETERS:")
    print("-" * 50)
    param_descriptions = {
        'a': 'Scale factor',
        'b': 'Power exponent for msg_size',
        'c': 'Power exponent for p_send',
        'd': 'Power exponent for cluster_size'
    }
    
    for param_name, param_value in params.items():
        desc = param_descriptions.get(param_name, '')
        print(f"{param_name:>6} = {param_value:12.8f}  ({desc})")
    print()
    
    # Performance metrics
    print("PERFORMANCE METRICS:")
    print("-" * 50)
    print(f"R² Score              = {metrics['r2']:10.6f}")
    print(f"Mean Absolute Error   = {metrics['mae']:10.6f}")
    print(f"Root Mean Square Error= {metrics['rmse']:10.6f}")
    print(f"Mean Abs. Perc. Error = {metrics['mape']:10.2f}%")
    print(f"Huber Loss            = {metrics['huber_loss']:10.6f}")
    print(f"Median Absolute Error = {metrics['median_abs_error']:10.6f}")
    print()
    
    # Residual statistics
    print("RESIDUAL STATISTICS:")
    print("-" * 50)
    print(f"Kurtosis (heavy tails)= {metrics['kurtosis']:10.6f}  (0=normal, >0=heavy tails)")
    print(f"Skewness (asymmetry)  = {metrics['skewness']:10.6f}  (0=symmetric)")
    print(f"Standard deviation    = {np.std(metrics['residuals']):10.6f}")
    print()
    
    # Prediction analysis
    y_pred = df_results['predicted'].values
    print("PREDICTION ANALYSIS:")
    print("-" * 50)
    print(f"Prediction range: [{np.min(y_pred):8.6f}, {np.max(y_pred):8.6f}]")
    print(f"Actual range:     [{np.min(df_results['success_probability']):8.6f}, {np.max(df_results['success_probability']):8.6f}]")
    print()
    
    # Performance by message size
    print("PERFORMANCE BY MESSAGE SIZE:")
    print("-" * 50)
    print(f"{'msg_size':<10} {'Count':<7} {'MAE':<10} {'Median_AE':<12} {'Max_AE':<10}")
    print("-" * 50)
    
    for msg_size, group in df_results.groupby('msg_size'):
        mae_group = np.mean(group['absolute_error'])
        median_ae = np.median(group['absolute_error'])
        max_ae = np.max(group['absolute_error'])
        count = len(group)
        print(f"{int(msg_size):<10} {count:<7} {mae_group:<10.6f} {median_ae:<12.6f} {max_ae:<10.6f}")
    print()
    
    # Worst and best predictions
    print("WORST PREDICTIONS (Top 5 by absolute error):")
    print("-" * 80)
    worst_predictions = df_results.nlargest(5, 'absolute_error')[
        ['payload_size', 'neighbors_per_probe', 'p_send', 'success_probability', 'predicted', 'absolute_error']
    ]
    print(worst_predictions.to_string(index=False, float_format='%.6f'))
    print()
    
    print("BEST PREDICTIONS (Top 5 by smallest absolute error):")
    print("-" * 80)
    best_predictions = df_results.nsmallest(5, 'absolute_error')[
        ['payload_size', 'neighbors_per_probe', 'p_send', 'success_probability', 'predicted', 'absolute_error']
    ]
    print(best_predictions.to_string(index=False, float_format='%.6f'))

# =============================================================================
# PLOTTING FUNCTIONS (Individual PDF files)
# =============================================================================

def plot_predicted_vs_actual(df_results, params, metrics):
    """Plot 1: Model Performance - Predicted vs Actual"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by message size
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    msg_sizes = sorted(df_results['msg_size'].unique())
    
    for i, msg_size in enumerate(msg_sizes):
        subset = df_results[df_results['msg_size'] == msg_size]
        ax.scatter(subset['success_probability'], subset['predicted'], 
                  color=colors[i], alpha=0.8, s=100, 
                  label=f'msg_size = {int(msg_size)}', 
                  edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Perfect Prediction')
    
    # Formatting
    ax.set_xlabel('Actual Success Probability', fontsize=14)
    ax.set_ylabel('Predicted Success Probability', fontsize=14)
    ax.set_title('Performance: Predicted vs Actual\n' + 
                f'P_success = 1/(1 + a×msg^b×p^c×cluster^d)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add performance metrics
    textstr = f'R² = {metrics["r2"]:.4f}\nMAE = {metrics["mae"]:.4f}\nRMSE = {metrics["rmse"]:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('model_predicted_vs_actual.pdf', dpi=300, bbox_inches='tight')

def plot_residuals_comprehensive(df_results, metrics):
    """Plot 2: Comprehensive Residuals Analysis"""
    residuals = metrics['residuals']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Residuals Analysis', fontsize=16)
    
    # 1. Residuals vs predicted
    ax1 = axes[0, 0]
    ax1.scatter(df_results['predicted'], residuals, alpha=0.7, s=60, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Add LOWESS trend line
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(df_results['predicted'])
    sorted_pred = df_results['predicted'].iloc[sorted_idx]
    sorted_resid = residuals[sorted_idx]
    if len(sorted_resid) > 5:
        trend = savgol_filter(sorted_resid, min(len(sorted_resid)//3*2+1, 15), 2)
        ax1.plot(sorted_pred, trend, 'orange', linewidth=2, alpha=0.8, label='Trend')
        ax1.legend()
    
    # 2. Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot (Normal Distribution)\nKurtosis = {metrics["kurtosis"]:.2f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals distribution with fits
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, density=True, alpha=0.7, color='lightblue', 
             edgecolor='black', label='Observed')
    
    # Fit different distributions
    x = np.linspace(residuals.min(), residuals.max(), 100)
    
    # Normal distribution
    mu, sigma = stats.norm.fit(residuals)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
    
    # t-distribution (better for heavy tails)
    df_param, loc, scale = stats.t.fit(residuals)
    ax3.plot(x, stats.t.pdf(x, df_param, loc, scale), 'g-', lw=2, 
             label=f't-dist (df={df_param:.1f})')
    
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title('Residuals Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scale-Location plot (homoscedasticity check)
    ax4 = axes[1, 1]
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    ax4.scatter(df_results['predicted'], sqrt_abs_resid, alpha=0.7, s=60, color='purple')
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('√|Residuals|')
    ax4.set_title('Scale-Location Plot\n(Check for Homoscedasticity)')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    if len(sqrt_abs_resid) > 5:
        sorted_idx = np.argsort(df_results['predicted'])
        sorted_pred = df_results['predicted'].iloc[sorted_idx]
        sorted_sqrt_resid = sqrt_abs_resid[sorted_idx]
        trend = savgol_filter(sorted_sqrt_resid, min(len(sorted_sqrt_resid)//3*2+1, 15), 2)
        ax4.plot(sorted_pred, trend, 'orange', linewidth=2, alpha=0.8, label='Trend')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('model_residuals_analysis.pdf', dpi=300, bbox_inches='tight')

def plot_performance_by_conditions(df_results):
    """Plot 3: Model Performance by Experimental Conditions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance by Experimental Conditions', fontsize=16)
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    msg_sizes = sorted(df_results['msg_size'].unique())
    
    # 1. Error vs p_send by message size
    ax1 = axes[0, 0]
    for i, msg_size in enumerate(msg_sizes):
        subset = df_results[df_results['msg_size'] == msg_size]
        ax1.scatter(subset['p_send'], subset['absolute_error'], 
                   color=colors[i], alpha=0.7, s=80, label=f'msg_size = {int(msg_size)}')
        
        # Add trend line
        if len(subset) > 2:
            z = np.polyfit(subset['p_send'], subset['absolute_error'], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(subset['p_send'].min(), subset['p_send'].max(), 50)
            ax1.plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('p_send')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Error vs p_send (with trend lines)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot by cluster size
    ax2 = axes[0, 1]
    cluster_sizes = sorted(df_results['cluster_size'].unique())
    data_by_cluster = [df_results[df_results['cluster_size'] == cs]['absolute_error'] for cs in cluster_sizes]
    
    box_plot = ax2.boxplot(data_by_cluster, labels=[f'{int(cs)}' for cs in cluster_sizes], 
                          patch_artist=True, notch=True)
    
    # Color the boxes
    box_colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error Distribution by Cluster Size')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Heatmap of MAE by conditions
    ax3 = axes[1, 0]
    pivot_table = df_results.pivot_table(values='absolute_error', 
                                        index='msg_size', 
                                        columns='cluster_size', 
                                        aggfunc='mean')
    
    im = ax3.imshow(pivot_table.values, cmap='RdYlBu_r', aspect='auto', 
                   vmin=pivot_table.values.min(), vmax=pivot_table.values.max())
    ax3.grid(None)
    
    ax3.set_xticks(range(len(pivot_table.columns)))
    ax3.set_yticks(range(len(pivot_table.index)))
    ax3.set_xticklabels([f'{int(cs)}' for cs in pivot_table.columns])
    ax3.set_yticklabels([f'{int(ms)}' for ms in pivot_table.index])
    ax3.set_xlabel('Cluster Size')
    ax3.set_ylabel('Message Size')
    ax3.set_title('Mean Absolute Error Heatmap')
    
    # Add colorbar and text annotations
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Mean Absolute Error')
    
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = ax3.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                           ha="center", va="center", 
                           color="white" if pivot_table.iloc[i, j] > pivot_table.values.mean() else "black",
                           fontweight='bold')
    
    # 4. Power law effects visualization
    ax4 = axes[1, 1]
    
    # Show how different terms scale with conditions
    for i, msg_size in enumerate(msg_sizes):
        subset = df_results[df_results['msg_size'] == msg_size]
        # Use fitted parameters for visualization
        a, b, c, d = 0.001, 1.5, 2.0, 1.0  # Example values
        term_contribution = a * (subset['msg_size'] ** b) * (subset['p_send'] ** c) * (subset['cluster_size'] ** d)
        ax4.scatter(term_contribution, subset['absolute_error'], 
                   color=colors[i], alpha=0.7, s=80, label=f'msg_size = {int(msg_size)}')
    
    ax4.set_xlabel('Model Term (a × msg^b × p^c × cluster^d)')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Model Error vs Power Law Term')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance_by_conditions.pdf', dpi=300, bbox_inches='tight')

def plot_model_behavior_3d(params, df_results):
    """Plot 4: 3D Model Behavior Visualization"""
    fig = plt.figure(figsize=(20, 6))
    
    # Create parameter ranges for prediction
    p_send_range = np.linspace(0.1, 1.0, 30)
    msg_sizes = sorted(df_results['msg_size'].unique())
    cluster_sizes = sorted(df_results['cluster_size'].unique())
    
    for idx, msg_size in enumerate(msg_sizes):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        # Create meshgrid
        P, C = np.meshgrid(p_send_range, cluster_sizes)
        Z = np.zeros_like(P)
        
        # Compute predictions
        for i, cluster_size in enumerate(cluster_sizes):
            for j, p_send in enumerate(p_send_range):
                Z[i, j] = model_simple(params, msg_size, cluster_size, p_send)
        
        # Plot surface with transparency
        surf = ax.plot_surface(P, C, Z, cmap='viridis', alpha=0.7, 
                              linewidth=0, antialiased=True)
        
        # Add actual data points
        subset = df_results[df_results['msg_size'] == msg_size]
        ax.scatter(subset['p_send'], subset['cluster_size'], subset['success_probability'], 
                  color='red', s=70, alpha=0.8, label='Actual Data', edgecolors='black')
        
        # Add predicted points for comparison
        ax.scatter(subset['p_send'], subset['cluster_size'], subset['predicted'], 
                  color='yellow', s=100, alpha=0.9, label='Model Predictions', 
                  marker='^', edgecolors='black')
        
        ax.set_xlabel('p_send', fontsize=12)
        ax.set_ylabel('Cluster Size', fontsize=12)
        ax.set_zlabel('Success Probability', fontsize=12)
        ax.set_title(f'Model Behavior\n(msg_size = {int(msg_size)})', fontsize=14)
        ax.legend(loc='upper left')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.savefig('model_behavior_3d.pdf', dpi=300, bbox_inches='tight')

def plot_convergence_analysis(convergence_history, params):
    """Plot 5: Optimization Convergence and Parameter Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimization Analysis', fontsize=16)
    
    # 1. Convergence plot
    ax1 = axes[0, 0]
    iterations = range(1, len(convergence_history) + 1)
    ax1.plot(iterations, convergence_history, linewidth=2, color='blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Huber Loss')
    ax1.set_title('CMA-ES Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add final value annotation
    final_loss = convergence_history[-1]
    ax1.annotate(f'Final: {final_loss:.6f}', 
                xy=(len(convergence_history), final_loss),
                xytext=(len(convergence_history)*0.7, final_loss*2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # 2. Parameter values with descriptions
    ax2 = axes[0, 1]
    param_names = list(params.keys())
    param_values = list(params.values())
    param_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax2.bar(param_names, param_values, color=param_colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Fitted Parameter Values')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')  # Log scale for better visibility
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Parameter correlation/sensitivity analysis
    ax3 = axes[1, 0]
    
    # Simulate parameter sensitivity
    df_data = load_data()
    base_prediction = model_simple(params, df_data['msg_size'], 
                                       df_data['cluster_size'], df_data['p_send'])
    
    sensitivities = {}
    perturbation = 0.1  # 10% perturbation
    
    for param_name in params.keys():
        perturbed_params = params.copy()
        perturbed_params[param_name] *= (1 + perturbation)
        perturbed_prediction = model_simple(perturbed_params, df_data['msg_size'], 
                                                df_data['cluster_size'], df_data['p_send'])
        sensitivity = np.mean(np.abs(perturbed_prediction - base_prediction))
        sensitivities[param_name] = sensitivity
    
    sens_names = list(sensitivities.keys())
    sens_values = list(sensitivities.values())
    bars3 = ax3.bar(sens_names, sens_values, color=param_colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Mean Absolute Change in Prediction')
    ax3.set_title('Parameter Sensitivity (10% perturbation)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # 4. Model term contributions
    ax4 = axes[1, 1]
    
    # Calculate different terms
    base_term = np.ones(len(df_data))
    power_term = params['a'] * (df_data['msg_size'] ** params['b']) * \
                (df_data['p_send'] ** params['c']) * \
                (df_data['cluster_size'] ** params['d'])
    
    # Pie chart of average contributions
    contributions = [
        np.mean(base_term),
        np.mean(power_term)
    ]
    labels = ['1 (base)', 'a×msg^b×p^c×cluster^d']
    colors_pie = ['lightblue', 'lightgreen']
    
    wedges, texts, autotexts = ax4.pie(contributions, labels=labels, colors=colors_pie, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('Average Model Term Contributions\n(to denominator)')
    
    plt.tight_layout()
    plt.savefig('model_convergence_analysis.pdf', dpi=300, bbox_inches='tight')

def plot_power_law_effects(df_results, params):
    """Plot 6: Power Law Effects Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Power Law Effects Analysis', fontsize=16)
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    msg_sizes = sorted(df_results['msg_size'].unique())
    
    # 1. Effect of msg_size power (b parameter)
    ax1 = axes[0, 0]
    msg_range = np.linspace(6, 103, 50)
    for i, cluster_size in enumerate([2, 4, 7]):
        for p_send in [0.2, 0.5, 0.8]:
            effect = msg_range ** params['b']
            ax1.plot(msg_range, effect, color=colors[i], 
                    alpha=0.7, linestyle=['--', '-', ':'][int((p_send-0.2)/0.3)],
                    label=f'cluster={cluster_size}, p={p_send}' if i == 0 else '')
    
    ax1.set_xlabel('Message Size')
    ax1.set_ylabel(f'msg_size^{params["b"]:.2f}')
    ax1.set_title(f'Message Size Power Effect (b={params["b"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Effect of p_send power (c parameter)
    ax2 = axes[0, 1]
    p_range = np.linspace(0.1, 1.0, 50)
    for i, msg_size in enumerate(msg_sizes):
        effect = p_range ** params['c']
        ax2.plot(p_range, effect, color=colors[i], linewidth=2,
                label=f'msg_size = {int(msg_size)}')
    
    # Overlay actual data points
    ax2.scatter(df_results['p_send'], df_results['p_send'] ** params['c'], 
               c='red', s=50, alpha=0.6, label='Data points')
    
    ax2.set_xlabel('p_send')
    ax2.set_ylabel(f'p_send^{params["c"]:.2f}')
    ax2.set_title(f'p_send Power Effect (c={params["c"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Effect of cluster_size power (d parameter)
    ax3 = axes[1, 0]
    cluster_range = np.linspace(2, 7, 50)
    for i, msg_size in enumerate(msg_sizes):
        effect = cluster_range ** params['d']
        ax3.plot(cluster_range, effect, color=colors[i], linewidth=2,
                label=f'msg_size = {int(msg_size)}')
    
    # Overlay actual data points
    ax3.scatter(df_results['cluster_size'], df_results['cluster_size'] ** params['d'], 
               c='red', s=50, alpha=0.6, label='Data points')
    
    ax3.set_xlabel('Cluster Size')
    ax3.set_ylabel(f'cluster_size^{params["d"]:.2f}')
    ax3.set_title(f'Cluster Size Power Effect (d={params["d"]:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Combined power law term vs success probability
    ax4 = axes[1, 1]
    
    # Calculate the full power law term
    power_term = params['a'] * (df_results['msg_size'] ** params['b']) * \
                 (df_results['p_send'] ** params['c']) * \
                 (df_results['cluster_size'] ** params['d'])
    
    for i, msg_size in enumerate(msg_sizes):
        subset_idx = df_results['msg_size'] == msg_size
        ax4.scatter(power_term[subset_idx], df_results.loc[subset_idx, 'success_probability'], 
                   color=colors[i], alpha=0.7, s=80, label=f'msg_size = {int(msg_size)}')
    
    # Show theoretical curve
    term_range = np.logspace(np.log10(power_term.min()), np.log10(power_term.max()), 100)
    theoretical_prob = 1.0 / (1.0 + term_range)
    ax4.plot(term_range, theoretical_prob, 'k--', linewidth=2, 
             alpha=0.8, label='Theoretical: 1/(1+term)')
    
    ax4.set_xlabel('Power Law Term (a×msg^b×p^c×cluster^d)')
    ax4.set_ylabel('Success Probability')
    ax4.set_title('Power Law Term vs Success Probability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('model_power_law_effects.pdf', dpi=300, bbox_inches='tight')

def main():
    """Main execution function"""
    print("SIMPLE POWER LAW MODEL")
    print("P_success = 1/(1 + a × msg_size^b × p_send^c × cluster_size^d)")
    print("Parameters: 4 (a, b, c, d)")
    print("Optimization: CMA-ES with increased budget")
    print("Loss Function: Huber Loss (robust to outliers)")
    print("="*80)
    
    # Fit the model with increased budget
    try:
        params, metrics, df_results, convergence_history = fit_model_robust(max_iter=8000, seed=42)
        
        # Print detailed results
        print_detailed_results(params, metrics, df_results)
        
        # Generate all plots as individual PDFs
        print("\nGenerating comprehensive analysis plots...")
        
        plot_predicted_vs_actual(df_results, params, metrics)
        plot_residuals_comprehensive(df_results, metrics)
        plot_performance_by_conditions(df_results)
        plot_model_behavior_3d(params, df_results)
        plot_convergence_analysis(convergence_history, params)
        plot_power_law_effects(df_results, params)
        
        print("\nAll plots saved as individual PDF files:")
        print("  - model_predicted_vs_actual.pdf")
        print("  - model_residuals_analysis.pdf")
        print("  - model_performance_by_conditions.pdf")
        print("  - model_behavior_3d.pdf")
        print("  - model_convergence_analysis.pdf")
        print("  - model_power_law_effects.pdf")
        
        # Save comprehensive results
        results_dict = {
            'model_name': 'Simple Power Law Model',
            'model_formula': '1/(1 + a × msg_size^b × p_send^c × cluster_size^d)',
            'parameters': {k: float(v) for k, v in params.items()},
            'metrics': {k: float(v) if k != 'residuals' else v.tolist() for k, v in metrics.items()},
            'num_parameters': 4,
            'num_data_points': len(df_results),
            'optimization_iterations': len(convergence_history),
            'key_features': [
                'Simple power law form with 4 parameters',
                'Captures non-linear scaling with all variables',
                'Interpretable parameter meanings', 
                'Robust optimization with Huber loss'
            ]
        }
        
        with open('model_simple_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nComprehensive results saved to: model_simple_results.json")
        
        # Summary
        print(f"\nMODEL SUMMARY:")
        print(f"{'='*40}")
        print(f"RÂ² Score:                {metrics['r2']:.4f}")
        print(f"Mean Absolute Error:     {metrics['mae']:.4f}")
        print(f"Parameters Used:         4")
        print(f"Model Form:              Simple Power Law")
        
    except Exception as e:
        print(f"Error during model fitting: {e}")
        raise

if __name__ == "__main__":
    main()
