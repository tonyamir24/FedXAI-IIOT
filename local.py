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
client_data = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9]

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



# ========== Set Random Seed ==========
seed = 42
np.random.seed(seed)
random.seed(seed)

# MLP Hyperparameters
hidden_layer_sizes = (20, 20)
batch_size = 2000
learning_rate = 0.001


# ========== Save models here ==========
models = {}
# Create global test set 
X_ext = pd.concat([client_test_data[j][0] for j in range(10) ], ignore_index=True)
y_ext = pd.concat([client_test_data[j][1] for j in range(10) ], ignore_index=True)

performance_local = []
performance_global = []

# ========== Train & Evaluate for Each Client ==========
for client_idx in range(10):
    

    X_train= client_train_data[client_idx][0]
    y_train = client_train_data[client_idx][1]
    X_test = client_test_data[client_idx][0]
    y_test = client_test_data[client_idx][1]

    

    print(f"\n================ Client {client_idx} ================")
    
    # Initialize model for this client
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        verbose=False,
        random_state=seed
    )
    
    # Train model (11 epochs)
    train_time = time.time()
    for epoch in range(11):
        model.fit(X_train, y_train)

    # Save model to dictionary
    models[client_idx] = model

    # -------- Local Test Evaluation --------
    y_pred_local = model.predict(X_test)
    acc_local = accuracy_score(y_test, y_pred_local)
    prec_local = precision_score(y_test, y_pred_local, average="weighted")
    rec_local = recall_score(y_test, y_pred_local, average="weighted")
    f1_local = f1_score(y_test, y_pred_local, average="weighted")

    print("\n--- Local Test Set ---")
    print(f"Accuracy: {acc_local:.4f}, Precision: {prec_local:.4f}, Recall: {rec_local:.4f}, F1: {f1_local:.4f}")
    performance_local.append({
        "Client": f"Client {client_idx}",
        "Accuracy": acc_local,
        "Precision": prec_local,
        "Recall": rec_local,
        "F1-Score": f1_local,
        "Training Time": time.time() - train_time
    })
    # -------- Global/External Test Evaluation --------
    y_pred_global = model.predict(X_ext)
    acc_global = accuracy_score(y_ext, y_pred_global)
    prec_global = precision_score(y_ext, y_pred_global, average="weighted")
    rec_global = recall_score(y_ext, y_pred_global, average="weighted")
    f1_global = f1_score(y_ext, y_pred_global, average="weighted")
    performance_global.append({
            "Client": f"Client {client_idx}",
            "Accuracy": acc_global,
            "Precision": prec_global,
            "Recall": rec_global,
            "F1-Score": f1_global,
            "Training Time": time.time() - train_time
        })
    
    print("\n--- Global Test Set (other clients) ---")
    print(f"Accuracy: {acc_global:.4f}, Precision: {prec_global:.4f}, Recall: {rec_global:.4f}, F1: {f1_global:.4f}")
    print(f"Training time: {time.time() - train_time:.2f} seconds")



final = {}
for client_idx in range(10):
    print(f"\n================================ Client {client_idx} Model Summary ================================")
    X_train= client_train_data[client_idx][0]
    y_train = client_train_data[client_idx][1]
    X_test = client_test_data[client_idx][0]
    y_test = client_test_data[client_idx][1]
    model  = models[client_idx]

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
    final[client_idx] = dict
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