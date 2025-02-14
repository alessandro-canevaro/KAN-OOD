import copy
from typing import Any
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import gaussian_kde

from source.efficient_kan import KAN
#from source.histograms import compute_binary_histograms, query_histograms


class KANBasePostprocessor():
    def __init__(self, params_config=None):        
        self.pc = self.default_config()
        self.configure(params_config)

        self.setup_flag = False

        # Define model for fttransf [-1, 3] else [-5, 5]
        self.kan_model_untrained = KAN([self.pc["kan_num_inputs"], self.pc["num_classes"]], grid_range=[0, 1] if self.pc["norm"] else [-1, 3], grid_size=self.pc["grid_size"]).to(self.pc["device"])
        self.kan_models = [copy.deepcopy(self.kan_model_untrained) for _ in range(self.pc["num_partitions"])]

    @classmethod
    def default_config(cls):
        """
        Default environment configuration.
        Can be overloaded in environment implementations, or by calling configure().
        Returns:
            dict: A configuration dictionary.
        """
        return {
            "num_classes": 10,
            "grid_size": 50,
            "num_partitions": 10,
            "norm": 1,
            "device": "cpu",
            "mode": 1,
            "kan_num_inputs": 512,
            "lr": 0.1,
            "epochs": 1,
            "batch_size": 512,
            "train_size": 50000,
            "pca_comp": 16,
            "hist_bins": 15,
        }

    def normalize_data(self, data, cdf_normalized_list, bin_edges_list):
        normalized_data = torch.zeros_like(data)
        num_features = data.shape[1]
        
        for feature_idx in range(num_features):
            feature_data = data[:, feature_idx].detach().cpu().numpy()
            cdf_normalized = cdf_normalized_list[feature_idx]
            bin_edges = bin_edges_list[feature_idx]
            normalized_feature = np.interp(feature_data, bin_edges[:-1], cdf_normalized)
            normalized_data[:, feature_idx] = torch.tensor(normalized_feature)
        
        return normalized_data

    
    def kan_setup(self, all_feats, all_labels_reduced, all_labels):
        print("Data prep")
        torch.autograd.set_detect_anomaly(True)
        all_feats = all_feats.to(self.pc["device"])
        all_labels = all_labels.to(self.pc["device"])
        all_labels_reduced = all_labels_reduced.to(self.pc["device"])
        print("dataset shape", all_feats.shape)

        if self.pc["norm"]:
            cdf_normalized_list = []
            bin_edges_list = []
   
            for feature_idx in range(self.pc["kan_num_inputs"]):
                feature_data = all_feats[:, feature_idx].detach().cpu().numpy()
                hist, bin_edges = np.histogram(feature_data, bins=self.pc["hist_bins"], range=(feature_data.min(), feature_data.max()), density=True)
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]  # Normalize to range [0, 1]
                
                cdf_normalized_list.append(cdf_normalized)
                bin_edges_list.append(bin_edges)
            # Save the CDF and bin edges for each feature
            np.save('cdf_normalized_list.npy', cdf_normalized_list)
            np.save('bin_edges_list.npy', bin_edges_list)
            all_feats = self.normalize_data(all_feats, cdf_normalized_list, bin_edges_list)
            print("max, min", torch.max(all_feats), torch.min(all_feats))
                

        print("training model")

        if self.pc["mode"] == 0: #class based
            num_samples = all_feats.shape[0] // len(self.kan_models)
            partitioned_data = [(all_feats[i * num_samples:(i + 1) * num_samples, :], 
                               all_labels[i * num_samples:(i + 1) * num_samples]) for i in range(len(self.kan_models))]
            # try:
            #     all_labels = all_labels // int((torch.max(all_labels).item() + 1)/self.pc["num_partitions"])
            # except RuntimeError:
            #     print("ZeroDivisionError")

            # unique_labels = np.unique(all_labels.cpu().detach().numpy())
            # partitioned_data = []
            # for label in unique_labels:
            #     # Get the indices of samples with the current label
            #     indices = np.where(all_labels.cpu().detach().numpy() == label)[0]
            #     # Partition all_feats and all_labels based on these indices
            #     feats_partition = all_feats[indices, :]
            #     labels_partition = all_labels_reduced[indices]
            #     partitioned_data.append((feats_partition, labels_partition))
        elif self.pc["mode"] == 1:# pca+kmeans
            # Step 1: Apply PCA to the dataset
            pca = PCA(n_components=self.pc["pca_comp"])
            principal_components = pca.fit_transform(all_feats.cpu())
            
            # Step 2: Perform KMeans clustering on the PCA features
            kmeans = KMeans(n_clusters=len(self.kan_models))
            cluster_assignments = kmeans.fit_predict(principal_components)

            # Step 3: Partition the data based on the cluster assignments
            partitioned_data = []
            for cluster in range(len(self.kan_models)):
                cluster_indices = (cluster_assignments == cluster)
                partition_feats = all_feats[cluster_indices]
                partition_labels = all_labels[cluster_indices]
                partitioned_data.append((partition_feats, partition_labels))
        else:
            raise ValueError("Invalid mode")
            
            

        # Train the models on each partition
        for i, (model, (partition_feats, partition_labels)) in enumerate(zip(self.kan_models, partitioned_data)):
            torch_dataset = torch.utils.data.TensorDataset(partition_feats.to(self.pc["device"]), 
                                                            partition_labels.to(self.pc["device"]))
            train_dataloader = torch.utils.data.DataLoader(torch_dataset, shuffle=True, batch_size=partition_feats.shape[0])
            self.train_kan(model, train_dataloader)

    def train_kan(self, model, train_dataloader):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.pc["lr"])
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.pc["epochs"]):
            model.train()
            for i, (data, label) in enumerate(train_dataloader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()    

    def compute_ood_score(self, data):
        if self.pc["norm"]:
            cdf_normalized_list = np.load('cdf_normalized_list.npy', allow_pickle=True)
            bin_edges_list = np.load('bin_edges_list.npy', allow_pickle=True)
            data = self.normalize_data(data, cdf_normalized_list, bin_edges_list)

        models_scores = []
        data = data.to(self.pc["device"])
        __ = self.kan_model_untrained(data)
        layer_0_untrained = self.kan_model_untrained.spline_layer[0].detach().cpu().numpy()
        for model in self.kan_models:
            out = model(data)
            layer_0_trained = model.spline_layer[0].detach().cpu().numpy()
            diff = np.abs(layer_0_trained[:, :, :] - layer_0_untrained[:, :, :])
            
            models_scores.append(np.median(diff, axis=(1, 2)))

        scores = np.max(models_scores, axis=0)
        return scores
    

    @staticmethod
    def _recursive_update(d, u):
        """
        Recursively update dictionary `d` with values from dictionary `u`.
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = KANBasePostprocessor._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def configure(self, config) -> None:
        """
        Configure.
        Args:
            config (dict): Configuration parameters.
        """
        if config:
            self.pc = self._recursive_update(self.pc, config)
