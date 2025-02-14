# ------------------------------------------------------------------------
# Advancing Out-of-Distribution Detection via Local Neuroplasticity
# Copyright (c) 2024 Alessandro Canevaro. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OpenOOD (https://github.com/Jingkang50/OpenOOD)
# Copyright (c) 2021 Jingkang Yang. All Rights Reserved.
# ------------------------------------------------------------------------

from copy import deepcopy
from typing import Any
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import numpy as np
import sklearn
import torch
import torch.nn as nn

from openood.postprocessors.base_postprocessor import BasePostprocessor
from openood.postprocessors.info import num_classes_dict

from source.postprocessor import KANBasePostprocessor


class KANPostprocessor(BasePostprocessor):
    def __init__(self, config, params_config=None):
        self.config = config
        self.dataset_name = self.config["dataset"]["name"]
        self.setup_flag = False

        self.args = self.config.postprocessor.postprocessor_args
        
        self.grid_size = self.args.grid_size
        self.num_partitions = self.args.num_partitions
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.pc = self.default_config()        

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
            "device": "cuda",
            "mode": 1,
            "pca_comp": 16,
            "hist_bins": 15,
            "aggregate_layers": 1,
            "train_size": 50000,
        }
        
    def _make_kan(self):
        self.pc = self.default_config()
        params_config = {"grid_size": self.grid_size,
                         "num_partitions": self.num_partitions,
                         "norm": 1,
                         "mode": get_preprocessing_info("mode", self.dataset_name), 
                         "hist_bins": get_preprocessing_info("hist_bins", self.dataset_name),
                         "num_classes": get_preprocessing_info("num_classes", self.dataset_name)}
        self.configure(params_config)
        # Define model
        self.pc["kan_num_inputs"] = sum([64, 64, 128, 256, 512][self.pc["aggregate_layers"]:])
        self.kan_postprocessor = KANBasePostprocessor(params_config=self.pc)

    def get_dataloader(self, id_loader_dict):
        from copy import deepcopy
        all_data = []
        all_labels = []
        with torch.no_grad():
            for batch in id_loader_dict['train']:
                data, labels = batch['data'], batch['label']
                
                all_data.append(deepcopy(data))
                all_labels.append(deepcopy(labels))

            all_data = torch.cat(all_data)
            all_labels = torch.cat(all_labels)
        # Initialize lists to store the reduced data and labels
        reduced_data = []
        reduced_labels = []
        for class_idx in range(10):
            class_indices = (all_labels == class_idx).nonzero(as_tuple=True)[0]
            if True:#class_idx < 5:
                selected_indices = class_indices[torch.randperm(len(class_indices))[:5000]]
            else:
                selected_indices = class_indices
            reduced_data.append(all_data[selected_indices])
            reduced_labels.append(all_labels[selected_indices])
        reduced_data = torch.cat(reduced_data, dim=0)
        reduced_labels = torch.cat(reduced_labels, dim=0)
        # Verify the shape of the reduced dataset
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(reduced_data, reduced_labels)
        dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
        return dataloader

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.dataset_name == "cifar10": #reduced training size experiment
            dataloader = self.get_dataloader(id_loader_dict)

        if not self.setup_flag:
            self._make_kan()


            all_feats = []
            all_labels = []
            all_preds = []

            with torch.no_grad():
                if self.dataset_name == "cifar10":
                    for batch in dataloader:
                        data, labels = batch
                        data = data.cuda()
                        logits, features_list = net(data, return_feature_list=True)
                        features = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[self.pc["aggregate_layers"]:]], dim=1)
    
                        all_feats.append(features.cpu())
                        all_labels.append(deepcopy(labels))
                        all_preds.append(logits.argmax(1).cpu())
                else:
                    for batch in id_loader_dict['train']:
                        data, labels = batch['data'].cuda(), batch['label']
                        logits, features = net(data, return_feature=True)
                        logits, features_list = net(data, return_feature_list=True)
                        features = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[self.pc["aggregate_layers"]:]], dim=1)
    
                        all_feats.append(features.cpu())
                        all_labels.append(deepcopy(labels))
                        all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            #all_feats = all_feats.view(all_feats.shape[0], all_feats.shape[1]//4, 4)
            #all_feats = all_feats.mean(dim=2)
          
            try:
                all_labels = all_labels // int((torch.max(all_labels).item() + 1)/self.pc["num_classes"])
                #all_labels_reduced = all_labels // int((torch.max(all_labels).item() + 1)/self.pc["num_classes"])
            except RuntimeError:
                print("ZeroDivisionError")

        

            self.kan_postprocessor.kan_setup(all_feats, all_labels)
            #self.kan_postprocessor.kan_setup(all_feats, all_labels_reduced, all_labels)
            
            self.setup_flag = True

            # Free GPU memory
            del data, labels, logits, features_list, features, all_feats, all_labels, all_preds
            torch.cuda.empty_cache()
            gc.collect()
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features_list = net(data, return_feature_list=True)
        features = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[self.pc["aggregate_layers"]:]], dim=1)

        #features = features.view(features.shape[0], features.shape[1]//4, 4)
        #features = features.mean(dim=2)

        pred = logits.argmax(1)
        
        scores = self.kan_postprocessor.compute_ood_score(features)

        conf = torch.tensor(scores).float() 
        return pred, conf
    
    @staticmethod
    def _recursive_update(d, u):
        """
        Recursively update dictionary `d` with values from dictionary `u`.
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = KANPostprocessor._recursive_update(d.get(k, {}), v)
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

    def set_hyperparam(self, hyperparam: list):
        self.grid_size = hyperparam[0]
        self.num_partitions = hyperparam[1]

    def get_hyperparam(self):
        return [self.grid_size,
                self.num_partitions]


def get_preprocessing_info(name, dataset):
    if name == "mode":
        return 1 if dataset == "cifar10" else 0
    if name == "hist_bins":
        return 5 if dataset == "cifar10" else 10
    if name == "num_classes":
        return 10 if dataset == "cifar10" else 20