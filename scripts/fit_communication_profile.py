#!/usr/bin/env python3
import numpy as np
import pandas as pd
from io import StringIO
from cma import CMAEvolutionStrategy
from sklearn.metrics import r2_score

# Load the message reception data
data = """payload_size,cluster_size,p_send,success_probability,std_deviation
3,18,0.1,0.998,0.0040
3,18,0.25,0.996,0.0112
3,18,0.5,0.993,0.0141
3,18,0.75,0.986,0.0179
3,18,0.9,0.967,0.0468
3,18,1,0.842,0.0999
3,4,0.1,0.998,0.0036
3,4,0.25,0.998,0.0036
3,4,0.5,0.996,0.0067
3,4,0.75,0.989,0.0179
3,4,0.9,0.993,0.0095
3,4,1,0.985,0.0260
3,2,0.1,0.986,0.0177
3,2,0.25,1.000,0.0000
3,2,0.5,0.995,0.0102
3,2,0.75,0.999,0.0030
3,2,0.9,1.000,0.0000
3,2,1,0.998,0.0036
20,18,0.1,0.997,0.0054
20,18,0.25,0.997,0.0043
20,18,0.5,0.882,0.1128
20,18,0.75,0.766,0.2111
20,18,0.9,0.570,0.3081
20,18,1,0.481,0.3092
20,4,0.1,0.989,0.0153
20,4,0.25,0.993,0.0085
20,4,0.5,0.994,0.0086
20,4,0.75,0.967,0.0388
20,4,0.9,0.775,0.1987
20,4,1,0.800,0.2340
20,2,0.1,0.992,0.0129
20,2,0.25,0.991,0.0151
20,2,0.5,0.992,0.0137
20,2,0.75,0.996,0.0080
20,2,0.9,0.997,0.0046
20,2,1,0.987,0.0231
100,18,0.01,0.976,0.0387
100,18,0.1,0.821,0.1679
100,18,0.25,0.395,0.3114
100,18,0.5,0.038,0.1089
100,18,0.75,0.018,0.0703
100,18,0.9,0.018,0.0454
100,18,1,0.025,0.0771
100,4,0.01,0.920,0.0872
100,4,0.1,0.929,0.0494
100,4,0.25,0.574,0.2788
100,4,0.5,0.166,0.1971
100,4,0.75,0.037,0.0645
100,4,0.9,0.030,0.0545
100,4,1,0.052,0.1012
100,2,0.01,0.912,0.1433
100,2,0.1,0.884,0.1164
100,2,0.25,0.815,0.1561
100,2,0.5,0.692,0.1971
100,2,0.75,0.634,0.1962
100,2,0.9,0.189,0.1629
100,2,1,0.131,0.1483"""

# Parse the data
df = pd.read_csv(StringIO(data))

# Add msg_size column (msg_size = 3 + payload_size + CRC + starting byte + ending byte)
df['msg_size'] = df['payload_size'] + 3 + 4 + 1 + 1

# Extract features and target
X = df[['msg_size', 'cluster_size', 'p_send']].values
y = df['success_probability'].values

# Model function: P(success) = 1 / (1 + (a * msg_size^b * p_send^c * cluster_size^d))
def model(params, X):
    a, b, c, d = params
    predictions = []
    
    for x in X:
        msg_size, cluster_size, p_send = x
        term = a * (msg_size ** b) * (p_send ** c) * (cluster_size ** d)
        prediction = 1 / (1 + term)
        predictions.append(prediction)
    
    return np.array(predictions)

# Objective function to maximize R²
def objective_function(params):
    # Ensure parameters are positive (except for potentially d)
    a, b, c, d = params
    a = abs(a)  # a should be positive
    
    # Predict using the model
    y_pred = model([a, b, c, d], X)
    
    # Calculate R² score (negative because we want to maximize)
    r2 = r2_score(y, y_pred)
    
    # Return negative R² (since CMA-ES minimizes)
    return -r2

# Calculate Mean Absolute Error for reporting
def calculate_mae(params, X, y):
    y_pred = model(params, X)
    return np.mean(np.abs(y - y_pred))

# Calculate MAE by msg_size
def calculate_mae_by_msg_size(params, df):
    results = {}
    for msg_size in df['msg_size'].unique():
        subset = df[df['msg_size'] == msg_size]
        X_subset = subset[['msg_size', 'cluster_size', 'p_send']].values
        y_subset = subset['success_probability'].values
        y_pred = model(params, X_subset)
        mae = np.mean(np.abs(y_subset - y_pred))
        results[msg_size] = mae
    return results

# Run CMA-ES optimization
def run_cmaes_optimization():
    # Initial parameters [a, b, c, d] and step size
    initial_params = [0.001, 1.5, 2.0, 1.0]
    sigma = 0.5  # step size
    
    # Create CMA-ES optimizer with constraints
    # Define lower and upper bounds separately
    lower_bounds = [1e-6, 0.1, 0.1, -2.0]
    upper_bounds = [10.0, 5.0, 5.0, 5.0]
    options = {
        'bounds': [lower_bounds, upper_bounds]
    }
    
    es = CMAEvolutionStrategy(initial_params, sigma, options)
    
    # Optimization loop
    print("Starting CMA-ES optimization...")
    print(f"Initial parameters: a={initial_params[0]}, b={initial_params[1]}, "
          f"c={initial_params[2]}, d={initial_params[3]}")
    
    iteration = 0
    best_params = initial_params
    best_score = objective_function(initial_params)
    
    while not es.stop():
        iteration += 1
        solutions = es.ask()
        function_values = [objective_function(x) for x in solutions]
        es.tell(solutions, function_values)
        
        # Update best parameters if better solution found
        current_best_idx = np.argmin(function_values)
        current_best_score = function_values[current_best_idx]
        
        if current_best_score < best_score:
            best_score = current_best_score
            best_params = solutions[current_best_idx]
            a, b, c, d = best_params
            r2 = -best_score  # Convert back to positive R²
            
            print(f"Iteration {iteration}: R² = {r2:.4f}, Parameters: a={a:.6f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
    
    return best_params, -best_score  # Return positive R²

# Run the optimization
best_params, best_r2 = run_cmaes_optimization()
a, b, c, d = best_params

# Calculate errors and evaluate final model
mae = calculate_mae(best_params, X, y)
mae_by_msg_size = calculate_mae_by_msg_size(best_params, df)

# Print final results
print("\n" + "="*50)
print("Final Model Results:")
print("="*50)
print(f"Best Parameters: a={a:.6f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
print(f"R² score: {best_r2:.4f}")
print(f"Overall Mean Absolute Error: {mae:.4f}")
print("\nMean Absolute Error by Message Size:")
for msg_size, error in mae_by_msg_size.items():
    print(f"  Message Size {msg_size}: {error:.4f}")

# Print the final model equation
print("\nFinal Model Equation:")
if d >= 0:
    print(f"P(success) = 1 / (1 + ({a:.6f} × msg_size^{b:.4f} × p_send^{c:.4f} × cluster_size^{d:.4f}))")
else:
    print(f"P(success) = 1 / (1 + ({a:.6f} × msg_size^{b:.4f} × p_send^{c:.4f} / cluster_size^{abs(d):.4f}))")

# Perform additional analysis by comparing predictions vs actual values
y_pred = model(best_params, X)
df_results = pd.DataFrame({
    'msg_size': df['msg_size'],
    'cluster_size': df['cluster_size'],
    'p_send': df['p_send'],
    'actual': df['success_probability'],
    'predicted': y_pred,
    'absolute_error': np.abs(df['success_probability'] - y_pred)
})

# Find the worst predicted cases
print("\nTop 5 Worst Predictions:")
worst_predictions = df_results.sort_values('absolute_error', ascending=False).head(5)
for _, row in worst_predictions.iterrows():
    print(f"Msg Size: {row['msg_size']}, Cluster: {row['cluster_size']}, P_send: {row['p_send']:.2f}, "
          f"Actual: {row['actual']:.4f}, Predicted: {row['predicted']:.4f}, Error: {row['absolute_error']:.4f}")

# Group by msg_size and calculate average error
print("\nAverage Error by Message Size:")
msg_size_errors = df_results.groupby('msg_size')['absolute_error'].mean()
for msg_size, error in msg_size_errors.items():
    print(f"  Message Size {msg_size}: {error:.4f}")

# Group by cluster size and calculate average error
print("\nAverage Error by Cluster Size:")
cluster_errors = df_results.groupby('cluster_size')['absolute_error'].mean()
for cluster, error in cluster_errors.items():
    print(f"  Cluster {cluster}: {error:.4f}")

# Visualize model predictions (using matplotlib but commented out as we're generating code only)
"""
import matplotlib.pyplot as plt

# Create a figure with subplots for each payload size
plt.figure(figsize=(15, 12))
for i, payload in enumerate([3, 20, 100]):
    plt.subplot(3, 1, i+1)
    
    # Filter data for this payload
    df_payload = df_results[df_results['payload_size'] == payload]
    
    # Create scatter plots for each cluster size
    for cluster in [2, 4, 18]:
        df_cluster = df_payload[df_payload['cluster_size'] == cluster]
        df_cluster = df_cluster.sort_values('p_send')
        
        plt.scatter(df_cluster['p_send'], df_cluster['actual'], 
                    label=f'Actual (Cluster {cluster})', marker='o')
        plt.plot(df_cluster['p_send'], df_cluster['predicted'], 
                 label=f'Predicted (Cluster {cluster})', linestyle='--')
    
    plt.title(f'Payload Size = {payload}')
    plt.xlabel('Probability of Sending')
    plt.ylabel('Success Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)

plt.tight_layout()
plt.savefig('model_predictions.png')
"""

# Function to predict success probability for new data points
def predict_success_probability(msg_size, cluster_size, p_send):
    params = best_params
    term = params[0] * (msg_size ** params[1]) * (p_send ** params[2]) * (cluster_size ** params[3])
    return 1 / (1 + term)

# Example usage
print("\nExample predictions:")
test_cases = [
    (12, 2, 0.5),   # Small message (3+3+4+1+1), small cluster, medium p_send
    (29, 4, 0.75),  # Medium message (3+20+4+1+1), medium cluster, high p_send
    (109, 18, 0.25) # Large message (3+100+4+1+1), large cluster, medium p_send
]

for msg_size, cluster, p_send in test_cases:
    pred = predict_success_probability(msg_size, cluster, p_send)
    print(f"Msg Size: {msg_size}, Cluster: {cluster}, P_send: {p_send:.2f} → Success Probability: {pred:.4f}")

if __name__ == "__main__":
    # Run the main optimization
    pass

