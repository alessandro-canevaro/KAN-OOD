# ------------------------------------------------------------------------
# Advancing Out-of-Distribution Detection via Local Neuroplasticity
# Copyright (c) 2024 Alessandro Canevaro. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OpenOOD (https://github.com/Jingkang50/OpenOOD)
# Copyright (c) 2021 Jingkang Yang. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Callable, List, Type, Any
import numpy as np
from openood.evaluators.metrics import compute_all_metrics
from openood.evaluation_api.evaluator import Evaluator
from openood.postprocessors import BasePostprocessor
from torch.nn.modules import Module


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)



class EvaluatorWrapper(Evaluator):

    def __init__(self, 
                 net: Module, 
                 id_name: str, 
                 data_root: str = './data', 
                 config_root: str = './configs', 
                 preprocessor: Callable[..., Any] = None, 
                 postprocessor_name: str = None, 
                 postprocessor: type[BasePostprocessor] = None, 
                 batch_size: int = 200, 
                 shuffle: bool = False, 
                 num_workers: int = 4) -> None:
        
        self.postprocessor_name = postprocessor_name
        super().__init__(net, 
                         id_name,
                         data_root, 
                         config_root, 
                         preprocessor, 
                         postprocessor_name, 
                         postprocessor, 
                         batch_size, 
                         shuffle, 
                         num_workers)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            if self.postprocessor_name == "kan":
                seed_everything(42)
                self.postprocessor.setup_flag = False
                self.postprocessor.setup(self.net, self.dataloader_dict['id'], self.dataloader_dict['ood'])

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True