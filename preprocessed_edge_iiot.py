# imports
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

# Load the dataset
df = pd.read_csv('./dataset/DNN-EdgeIIoT-dataset.csv', low_memory=False)

# Drop unnecessary columns
drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 

         "http.file_data","http.request.full_uri","icmp.transmit_timestamp",

         "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",

         "tcp.dstport", "udp.port", "mqtt.msg", 'icmp.unused','dns.qry.type','mqtt.msg_decoded_as']

df.drop(drop_columns, axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df).reset_index(drop=True)

df.isna().sum()


print(f"Applying log transformation")
df_numeric = df.select_dtypes(include=[np.number])

for feature in df_numeric.columns:
   
    # Apply log transformation if the column has more than 50 unique values
    if df_numeric[feature].nunique() > 50:
        if df_numeric[feature].min() == 0:
            df[feature] = np.log(df[feature] + 1)
        else:
            df[feature] = np.log(df[feature])

print(f"Data after log transformation:\n", df.head())


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Attack_label')


from sklearn.preprocessing import StandardScaler

# Scale only numeric (continuous) features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df['http.referer'].value_counts()) #done
print(df['http.request.method'].value_counts()) #done
print(df['http.request.version'].value_counts()) #done
print(df['dns.qry.name.len'].value_counts()) #done
print(df['mqtt.conack.flags'].value_counts())  #done
print(df['mqtt.protoname'].value_counts()) #done
print(df['mqtt.topic'].value_counts()) #done



print(f"\nProcessing categorical columns ")
df_cat = df.select_dtypes(exclude=[np.number])

for feature in df_cat.columns:
    # Skip the attack_cat column (or attack_cat_encoded if numeric)
    if feature == 'Attack_type':
        continue
    
    # Set the unique count limit based on the feature
    if feature == 'http.request.method':
        unique_limit = 6
    elif feature == 'http.referer':
        unique_limit = 5
    elif feature == 'http.request.version':
        unique_limit = 4
    elif feature == 'dns.qry.name.len':
        unique_limit = 6
    elif feature == 'mqtt.conack.flags':
        unique_limit = 4
    elif feature == 'mqtt.protoname':
        unique_limit = 3  
    elif feature == 'mqtt.topic':
        unique_limit = 3
    else:
        continue  # Skip if it's not proto, service, or state
    
    # Count unique values
    unique_count = df_cat[feature].nunique()
    
    # Keep only the top categories if unique count exceeds the specified limit
    if unique_count > unique_limit:
        # Get the most frequent categories (e.g., top N based on unique_limit) and replace the rest with '-'
        top_categories = df_cat[feature].value_counts().nlargest(unique_limit - 1).index  # Adjust limit as needed
        df[feature] = np.where(
            df[feature].isin(top_categories),
            df[feature],
            '-'  # Replace with '-'
        )


df_cat = df.select_dtypes(exclude=[np.number])


print(f"\nTop categories ")

# Display top 6 categories for 'http.request.method'
unique_http_request_method = df['http.request.method'].value_counts().nlargest(6).index
unique_http_request_method_list = unique_http_request_method.tolist()
# Display top 6 categories for 'http.referer'
unique_http_referer = df['http.referer'].value_counts().nlargest(6).index
unique_http_referer_list = unique_http_referer.tolist()
# Display top 4 categories for 'http.request.version'   
unique_http_request_version = df['http.request.version'].value_counts().nlargest(4).index
unique_http_request_version_list = unique_http_request_version.tolist()
# Display top 6 categories for 'dns.qry.name.len'
unique_dns_qry_name_len = df['dns.qry.name.len'].value_counts().nlargest(6).index
unique_dns_qry_name_len_list = unique_dns_qry_name_len.tolist()
# Display top 4 categories for 'mqtt.conack.flags'      
unique_mqtt_conack_flags = df['mqtt.conack.flags'].value_counts().nlargest(4).index
unique_mqtt_conack_flags_list = unique_mqtt_conack_flags.tolist()
# Display top 3 categories for 'mqtt.protoname' 
unique_mqtt_protoname = df['mqtt.protoname'].value_counts().nlargest(3).index
unique_mqtt_protoname_list = unique_mqtt_protoname.tolist()
# Display top 3 categories for 'mqtt.topic'
unique_mqtt_topic = df['mqtt.topic'].value_counts().nlargest(3).index
unique_mqtt_topic_list = unique_mqtt_topic.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print(f"\nApplying one-hot encoding")

# Define which columns you want to encode
encoded_cols = ['http.request.method', 'http.referer', 'http.request.version',
                'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic']

# Define the ColumnTransformer with fixed categories for each column
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(categories=[
            unique_http_request_method_list,
            unique_http_referer_list,
            unique_http_request_version_list,
            unique_dns_qry_name_len_list,
            unique_mqtt_conack_flags_list,
            unique_mqtt_protoname_list,
            unique_mqtt_topic_list
        ], handle_unknown='ignore'), encoded_cols)
    ]
)

# Apply transformation
X_encoded = ct.fit_transform(df)

# Convert sparse matrix to dense array
X_encoded = X_encoded.toarray()

# Get feature names from encoder
feature_names = ct.get_feature_names_out()

# Create DataFrame from encoded features
df_encoded = pd.DataFrame(X_encoded, columns=feature_names, index=df.index)

# Drop the original categorical columns from df
df_other = df.drop(columns=encoded_cols)

# Combine the rest of the DataFrame with the encoded features
df = pd.concat([df_other, df_encoded], axis=1)

# Print the shape to confirm
print(f"Number of features : {df.shape[1]}")

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Attack_type'] = label_encoder.fit_transform(df['Attack_type'])

# Store class mapping (optional)
attack_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", attack_mapping)

# Split the data (80% train, 20% test)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['Attack_type'])

# Save train and test sets separately
df_train.to_csv('./dataset/train_edge_iiot.csv', index=False)
df_test.to_csv('./dataset/test_edge_iiot.csv', index=False)