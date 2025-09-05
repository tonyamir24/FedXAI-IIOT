# Explainable Heterogeneous Federated Learning for Intrusion Detection in IIoT

This repository contains the code and experiments for my Master's thesis project:  
**"	Explainable Heterogeneous Federated Learning for Intrusion Detection in IIoT"**.  

The project investigates how **Explainable AI (XAI)** techniques, such as **SHAP** and **LIME**, can be applied to interpret Intrusion Detection Systems (IDS) trained in **Industrial IoT (IIoT)** environments across three paradigms:
- **Centralized learning**
- **Federated learning**
- **Decentralized (local client) learning**

---

## 📂 Repository Structure

- **`xaimulticlass_centerlized.ipynb`**  
  Training and evaluation of the centralized IDS model with XAI analysis.

- **`xaimulticlass_fl.ipynb`**  
  Implementation of the federated learning setup, training across clients, and global aggregation with explainability.

- **`xaimulticlass_clients_local.ipynb`**  
  Training IDS models locally at client level (no aggregation), with XAI analysis for local interpretability.

- **`preprocessed_edge_iiot.py`**  
  Data preprocessing pipeline for the IIoT dataset, preparing features and labels for experiments.

- **`partioning_with_Dirichlet.ipynb`**  
  Dataset partitioning using **Dirichlet distribution**, simulating realistic non-IID client data distributions for federated learning.

- **`analysis_shape.ipynb`**  
  Exploratory analysis of SHAP values and visualization of feature importance across different learning paradigms.

---