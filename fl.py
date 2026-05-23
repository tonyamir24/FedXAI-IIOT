import numpy as np
import pandas as pd
import os
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import joblib

import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings
from IPython.display import display, HTML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import shap 
import lime# SHAP for explainability
from lime.lime_tabular import LimeTabularExplainer  # LIME for local explanations
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf


df0 = pd.read_csv('dataset/client_0_data.csv', low_memory=False)
df1 = pd.read_csv('dataset/client_1_data.csv', low_memory=False)
df2 = pd.read_csv('dataset/client_2_data.csv', low_memory=False)
df3 = pd.read_csv('dataset/client_3_data.csv', low_memory=False)
df4 = pd.read_csv('dataset/client_4_data.csv', low_memory=False)
df5 = pd.read_csv('dataset/client_5_data.csv', low_memory=False)
df6 = pd.read_csv('dataset/client_6_data.csv', low_memory=False)
df7 = pd.read_csv('dataset/client_7_data.csv', low_memory=False)
df8 = pd.read_csv('dataset/client_8_data.csv', low_memory=False)
df9 = pd.read_csv('dataset/client_9_data.csv', low_memory=False)

def drop_single_occurrence_rows(df, column):
    # Find values that occur more than once
    valid_values = df[column].value_counts()[lambda x: x > 1].index
    # Keep only rows with those values
    return df[df[column].isin(valid_values)]

# Example usage:
df0 = drop_single_occurrence_rows(df0, "Attack_type")
df1 = drop_single_occurrence_rows(df1, "Attack_type")
df2 = drop_single_occurrence_rows(df2, "Attack_type")
df3 = drop_single_occurrence_rows(df3, "Attack_type")
df4 = drop_single_occurrence_rows(df4, "Attack_type")
df5 = drop_single_occurrence_rows(df5, "Attack_type")
df6 = drop_single_occurrence_rows(df6, "Attack_type")
df7 = drop_single_occurrence_rows(df7, "Attack_type")
df8 = drop_single_occurrence_rows(df8, "Attack_type")
df9 = drop_single_occurrence_rows(df9, "Attack_type")



from sklearn.model_selection import train_test_split

# Set test size and random state for reproducibility
test_size = 0.2
random_state = 42

# List of all client dataframes
client_data = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9]  # Assuming df5 to df9 are defined similarly

# Prepare training and testing sets for each client
client_train_data = {}
client_test_data = {}

for i, df in enumerate(client_data):
    print(f"\nProcessing Client {i+1}")
    
    # Separate features and target
    X = df.drop(columns=['Attack_type', 'Attack_label'])  # Features
    y = df['Attack_type']

    # Split each client's data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)

    # Store the processed data for each client
    client_train_data[i] = (X_train, y_train)
    client_test_data[i] = (X_test, y_test)

    # Print confirmation
    print(f"Completed processing for Client {i+1}")
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")


# Initialize an empty DataFrame
model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'Precision', 'F1-Score', 
                                          'time to train', 'time to predict', 'total time'])


# Set Random Seed
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Configuration
num_rounds = 10 # Number of federated learning rounds
# epoch_schedule = [50, 50, 50, 50, 50, 50, 80, 80, 80, 80]  # Epochs per round
# epoch_schedule =[ 5, 5, 5, 5, 5, 2, 2, 2, 2, 1]  # Epochs per round
epoch_schedule =[ 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]  # Epochs per round
hidden_layer_sizes = (20, 20)  # Hidden layer configuration
batch_size = 2000  # Batch size for MLP
learning_rate = 0.001  # Learning rate for MLP

# Global list of all possible classes
all_classes = np.unique(np.concatenate([client_train_data[i][1] for i in range(len(client_train_data))]))



# Initialize global model parameters
global_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                             activation='relu', 
                             solver='adam', 
                             batch_size=batch_size, 
                             learning_rate_init=learning_rate, 
                             max_iter=1,  # One iteration per call to `fit`
                             warm_start=True, 
                             verbose=0)

# Perform a dummy fit to initialize the model structure with all classes
X_dummy = np.random.rand(15, client_train_data[0][0].shape[1])  # Random features
y_dummy = np.tile(all_classes, 1)[:15]  # Ensure all classes are present
global_model.fit(X_dummy, y_dummy)  # Dummy fit

# Assign global weights and biases from the initialized model
global_weights = copy.deepcopy(global_model.coefs_)
global_biases = copy.deepcopy(global_model.intercepts_)



# Function to aggregate weights using FedAvg
def fed_avg(weights_list, biases_list):
    avg_weights = [np.mean([w[layer] for w in weights_list], axis=0) for layer in range(len(global_weights))]
    avg_biases = [np.mean([b[layer] for b in biases_list], axis=0) for layer in range(len(global_biases))]
    return avg_weights, avg_biases


# Store performance metrics
round_performance = []
os.makedirs("saved_models", exist_ok=True)
# Federated Learning Rounds


# Initialize tracking variables
client_total_train_time = {f"Client {i}": 0.0 for i in range(len(client_train_data))}
round_times = []

# Federated Learning Rounds
models = {}
for round_num in range(1, num_rounds + 1):
    print(f"\n--- FL Round {round_num} ---")
    
    num_epochs = epoch_schedule[round_num - 1]
    print(f"Number of epochs for this round: {num_epochs}")
    
    round_start_time = time.time()
    client_weights = []
    client_biases = []
    
    # Local Training on Clients
    for i in range(len(client_train_data)):
        print(f"\nTraining model for Client {i}")
        X_train, y_train = client_train_data[i]
        X_test, y_test = client_test_data[i]

        # Add dummy samples if any classes are missing
        missing_classes = set(all_classes) - set(y_train)
        if missing_classes:
            print(f"  Adding dummy samples for missing classes: {missing_classes}")
            dummy_X = np.zeros((len(missing_classes), X_train.shape[1]))
            dummy_y = np.array(list(missing_classes))
            X_train = np.vstack([X_train, dummy_X])
            y_train = np.hstack([y_train, dummy_y])

        # Initialize local model with global weights
        local_model = copy.deepcopy(global_model)
        local_model.coefs_ = copy.deepcopy(global_weights)
        local_model.intercepts_ = copy.deepcopy(global_biases)

        client_start_time = time.time()

        # Train local model for specified epochs
        for epoch in range(num_epochs):
            local_model.fit(X_train, y_train)

        client_train_time = time.time() - client_start_time
        client_total_train_time[f"Client {i}"] += client_train_time

        # Store weights and biases
        client_weights.append(local_model.coefs_)
        client_biases.append(local_model.intercepts_)

        # Evaluate local model
        y_pred = local_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Client {i} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Train Time: {client_train_time:.4f} seconds")

        if round_num == num_rounds:
            models[f'Client {i}'] = local_model

        # Save performance metrics
        round_performance.append({
            "Round": round_num,
            "Client": f"Client {i}",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Train Time": client_train_time,
        })

    # End of round
    round_train_time = time.time() - round_start_time
    round_times.append(round_train_time)
    print(f"Total training time for Round {round_num}: {round_train_time:.4f} seconds")

    # FedAvg aggregation
    global_weights, global_biases = fed_avg(client_weights, client_biases)
    global_model.coefs_ = copy.deepcopy(global_weights)
    global_model.intercepts_ = copy.deepcopy(global_biases)

    if round_num == num_rounds:
        models['global'] = local_model



# === Performance Visualization ===
round_performance_df = pd.DataFrame(round_performance)
sns.lineplot(data=round_performance_df, x="Round", y="Accuracy", hue="Client")
plt.title("Accuracy Across Rounds (Federated MLP - Multiclass)")
plt.show()


X_test = pd.concat([client_test_data[j][0] for j in range(5)], ignore_index=True)
y_test = pd.concat([client_test_data[j][1] for j in range(5)], ignore_index=True)


import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# After all federated learning rounds have completed

# Set final global model weights
global_model.coefs_ = global_weights
global_model.intercepts_ = global_biases


# Predict with the global model and measure prediction time
start_time = time.time()
y_pred_global = global_model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred_global)
precision = precision_score(y_test, y_pred_global, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred_global, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_global, average='macro', zero_division=0)

cm_global = confusion_matrix(y_test, y_pred_global, labels=all_classes)

# Print metrics
print(f"\nConfusion Matrix for Global Model:\n{cm_global}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_global, display_labels=all_classes)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Global Model")
plt.show()



disp = ConfusionMatrixDisplay(confusion_matrix=cm_global, display_labels=all_classes)
fig, ax = plt.subplots(figsize=(12, 10))  # enlarge plot
disp.plot(cmap="Blues", ax=ax, values_format=".0f")  # ".0f" = no decimals
plt.title("Confusion Matrix - Global Model")
plt.xticks(rotation=45)  # rotate labels if long
plt.tight_layout()
plt.show()


X_train = pd.concat([client_train_data[j][0] for j in range(5)], ignore_index=True)
y_train = pd.concat([client_train_data[j][1] for j in range(5)], ignore_index=True)
X_test = pd.concat([client_test_data[j][0] for j in range(5)], ignore_index=True)
y_test = pd.concat([client_test_data[j][1] for j in range(5)], ignore_index=True)

model  = global_model
print(f"\nGenerating SHAP explanations for the final local models...")

X_explainer = X_test.values[:1000]
# X_explainer = X_test.values[:50]  # Subset of test data

# Select a representative background set (e.g., 100 samples)
background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train

# Initialize the SHAP KernelExplainer
explainer = shap.KernelExplainer(model.predict_proba, background)

# Compute SHAP values for the explanation set
shap_values = explainer.shap_values(X_explainer)

print(f"SHAP values shape: {[sv.shape for sv in shap_values] if isinstance(shap_values, list) else shap_values.shape}")
print(f"X_explainer shape: {X_explainer.shape}")
print(f"Number of classes: {len(model.classes_)}")

# # 1. Multi-class summary plot (shows all classes together)
# print("\n1. Generating multi-class summary plot...")
# shap.summary_plot(shap_values, X_explainer, feature_names=X_test.columns, 
#                   class_names=model.classes_)

# 2. Calculate feature importance percentages
print("\n2. Calculating feature importance percentages...")

# Handle SHAP values based on their structure
if isinstance(shap_values, list):
    # Multi-class case: shap_values is a list of arrays, one per class
    # Each array has shape (n_samples, n_features)
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif len(shap_values.shape) == 3:
    # Multi-class case: shap_values has shape (n_samples, n_features, n_classes)
    # Average across samples (axis=0) and classes (axis=2)
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    # Binary classification case: shape (n_samples, n_features)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

print(f"Mean absolute SHAP shape after processing: {mean_abs_shap.shape}")

# Calculate percentages
total_importance = np.sum(mean_abs_shap)
feature_importance_pct = (mean_abs_shap / total_importance) * 100

# Get feature names - use the actual column names from X_test
feature_names = list(X_test.columns)
n_features = len(mean_abs_shap)

print(f"Number of features: {n_features}")
print(f"Number of feature names: {len(feature_names)}")

# Ensure feature names match the number of features
if len(feature_names) != n_features:
    print(f"Error: Feature names length ({len(feature_names)}) doesn't match SHAP values length ({n_features})")
    # Use available feature names or create generic ones
    if len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
    else:
        feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])

# Create DataFrame for easier handling
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_abs_shap,
    'percentage': feature_importance_pct
}).sort_values('importance', ascending=False)

# Display top features with percentages
print("\nTop 10 Features by Importance:")
print("=" * 50)
dict = {}
for idx, row in importance_df.head(15).iterrows():
    dict[row['feature']]= round(row['percentage'], 2)
    print(f"{row['feature']:<30}: {row['percentage']:.2f}%")
# Display least important features
print("\n5 Least Important Features:")
print("=" * 50)
for idx, row in importance_df.tail(5).iterrows():
    print(f"{row['feature']:<30}: {row['percentage']:.2f}%")

# 3. Bar plot with percentages
print("\n3. Generating feature importance bar plot with percentages...")
plt.figure(figsize=(12, 8))

# Get top 15 features for plotting
top_features = importance_df.head(15)

# Create horizontal bar plot
bars = plt.barh(range(len(top_features)), top_features['percentage'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance (%)')
plt.title('Feature Importance Percentages (Based on Mean |SHAP Value|)')

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, top_features['percentage'])):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{pct:.1f}%', va='center', fontsize=9)

plt.gca().invert_yaxis()  # Highest importance at top
plt.tight_layout()
# plt.show()

# 4. Alternative: Use SHAP's built-in bar plot but create custom percentage display
print("\n4. Generating SHAP bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_explainer, plot_type="bar",
                feature_names=feature_names, class_names=model.classes_,
                max_display=15, show=False)
plt.title("Feature Importance Across All Classes (Mean |SHAP Value|)")

# Add percentage text to the current plot
ax = plt.gca()
bars = ax.patches
if len(bars) > 0:
    # Get the importance values from the bars
    bar_values = [bar.get_width() for bar in bars]
    total_bar_value = sum(bar_values)
    
#     # Add percentage labels
#     for bar in bars:
#         width = bar.get_width()
#         percentage = (width / total_bar_value) * 100
#         ax.text(width + max(bar_values) * 0.01, bar.get_y() + bar.get_height()/2,
#                 f'{percentage:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.show()

# 5. Print summary statistics
print(f"\nSummary:")
print(f"Total features: {len(importance_df)}")
print(f"Top 5 features account for: {importance_df.head(5)['percentage'].sum():.1f}% of total importance")
print(f"Top 10 features account for: {importance_df.head(10)['percentage'].sum():.1f}% of total importance")
print(f"Bottom 5 features account for: {importance_df.tail(5)['percentage'].sum():.1f}% of total importance")
print(f"Most important feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['percentage']:.2f}%)")
print(f"Least important feature: {importance_df.iloc[-1]['feature']} ({importance_df.iloc[-1]['percentage']:.2f}%)")