import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


import pandas as pd, ast



#########################################
# Part 1: Aggregating CSV Files
#########################################
T=5
# sample = 1000
# List of CSV files to aggregate
csv_files = [str(T)+"_4_node_simulation_data_million.csv",str(T)+"_6_node_simulation_data_million.csv",str(T)+"_8_node_simulation_data_million.csv",str(T)+"_10_node_simulation_data_million.csv",str(T)+"_12_node_simulation_data_million.csv"]

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    # Convert string representations to actual lists
    for col in ['adj_matrix', 'transmission_prob', 'saturation_throughput']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    dfs.append(df)

# Concatenate all DataFrames into one aggregated DataFrame
aggregated_df = pd.concat(dfs, ignore_index=True)
aggregated_csv = "Data/"+str(T)+"_data_million.csv"
aggregated_df.to_csv(aggregated_csv, index=False)
print("Aggregated dataset saved")

#########################################
# Part 2: Data Loading and Preprocessing
#########################################
