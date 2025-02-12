'''
Notes:
1. Install the following libraries before running codes
    !pip install git+https://github.com/Jingkang50/OpenOOD
    !pip install pytorch_lightning
    !pip install gpytorch
    !pip install --upgrade git+https://github.com/y0ast/DUE.git
    !pip install nflows

2. Place the preprocessed CSV files in the same directory as this file
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from reading_files.csv_read import read, check
from reading_files.feature_selection import get_eICU_selected_features, get_mimic_selected_features
from ood_experiment.experiments import get_params_data
from ood_experiment.validate_difference import validate_ood_data
from models.predictive_models import get_params
from training.utils import set_all_seeds
from training.data_handler import normalization, split_data
from training.train import train_predictive_model
from ood_measures.ood_score import eval_ood, get_ood_score, print_all_metrics
from ood_measures.detection_methods_posthoc import KANPostprocessor, MDSPostprocessor, detection_method



def get_ood_score_kan(model, in_test_features, in_test_labels, ood_type, score_function, batch_size, device, preprocess, random_sample=None, scale=None, out_features=None, missclass_as_ood=False):
    """
    Calculate the novelty scores that an OOD detector (score_function) assigns to ID and OOD and evaluate them via AUROC and FPR.

    Parameters:
    -----------
    model: torch.nn.Module or None
        The neural network model for applying the post-hoc method.
    in_test_features: torch.Tensor
        In-distribution test features.
    in_test_labels: torch.Tensor
        In-distribution test labels.
    ood_type: str
        The type of out-of-distribution (OOD) data ('other_domain', 'feature_separation', or 'multiplication').
    score_function: callable
        The scoring function that assigns each sample a novelty score.
    batch_size: int
        Batch size for processing data.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    preprocess: object
        The preprocess for normalizing the data if it is needed.
    random_sample: list or None, optional
        List of randomly selected feature indices for 'multiplication'. Default is None.
    scales: list or None, optional
        List of scales for feature multiplication. Default is None.
    out_features: torch.Tensor or None, optional
        Out-of-distribution (OOD) features for 'other_domain' or 'feature_separation'. Default is None.
    missclass_as_ood: bool, optional
        If True, consider misclassified in-distribution samples as OOD. Default is False.
    """
    
    if model is not None:
        model.eval() 
        
    preds_in, confs_in, gt_in = [], [], []
    for batch_idx in range(len(in_test_features) // batch_size):
        x_batch = in_test_features[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
        pred, conf = score_function(model, x_batch)
        preds_in += list(pred)
        confs_in += list(conf)
        gt_in += list(in_test_labels[batch_idx*batch_size:(batch_idx+1)*batch_size].cpu().detach().numpy())

    if ood_type in ['other_domain', 'feature_separation']:
        preds_out, confs_out, gt_out = [], [], []
        for batch_idx in range(len(out_features) // batch_size):
              x_batch = out_features[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
              pred, conf = score_function(model, x_batch)
              preds_out += list(pred)
              confs_out += list(conf)
              gt_out += list(np.ones(conf.shape[0])*-1)
        
        ood_metrics = eval_ood([preds_in, confs_in, preds_out, confs_out, gt_in, gt_out], to_print=False, missclass_as_ood=missclass_as_ood)
        #auc = ood_score_calc(scores_inlier, scores_ood)
        #print('AUC:', auc)
        print_all_metrics(ood_metrics)
        return ood_metrics[1] #0 FPR, 1 AUROC
    else:

        X_test_adjusted = torch.clone(in_test_features).cpu().numpy()
        X_test_adjusted = preprocess.inverse_transform(X_test_adjusted)

        scores_per_scale = np.zeros(5)
        X_test_adjusted_scaled = np.copy(X_test_adjusted)*scale
        X_test_adjusted_scaled = preprocess.transform(X_test_adjusted_scaled)

        for r in tqdm(random_sample):
            out_features = torch.clone(in_test_features).to(device)
            out_features[:,r] = torch.tensor(X_test_adjusted_scaled[:, r]).to(device)

            preds_out, confs_out, gt_out = [], [], []
            for batch_idx in range(len(out_features) // batch_size):
                    x_batch = out_features[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    pred, conf = score_function(model, x_batch)
                    preds_out += list(pred)
                    confs_out += list(conf)
                    gt_out += list(np.ones(conf.shape[0])*-1)
            
            ood_metrics = eval_ood([preds_in, confs_in, preds_out, confs_out, gt_in, gt_out], to_print=False, missclass_as_ood=missclass_as_ood)
            #print(ood_metrics)
            scores_per_scale += np.array(ood_metrics)
            #scores_per_scale += ood_score_calc(scores_inlier, scores_ood)
        
        print('Scale:', scale)
        print_all_metrics(list(scores_per_scale/len(random_sample)))
        #print('Scale:', scale, 'Average AUC:', scores_per_scale/len(random_sample))
        return list(scores_per_scale/len(random_sample))[1] #0 FPR, 1 AUROC

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
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_version', default='iv', type=str, choices={'iv', 'iii'})
    parser.add_argument('--in_distribution', default='mimic', type=str, choices={'mimic', 'eicu'})

    parser.add_argument('--ood_type', default='other_domain', type=str, choices={'other_domain', 'multiplication', 'feature_separation'})
    parser.add_argument('--feature_to_seperate', default='age', type=str, choices={'age', 'gender', 'ethnicity', 'admission_type', 'first_careunit'}) #only for 'feature_separation'
    parser.add_argument('--threshold', default='70', type=str) #threshold for dividing data in 'feature_separation'
    parser.add_argument('--scale', default='1000', type=str)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--d_out', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--handle_imbalance_data', default=0, type=int, choices={0, 1})
    parser.add_argument('--train_model', default=1, type=int, choices={0, 1})
    parser.add_argument('--architecture', default='ResNet', type=str, choices={'MLP', 'ResNet', 'FTTransformer'})

    parser.add_argument("--detectors", nargs='+', default=['KAN', 'MDS'])
    parser.add_argument('--grid_size', default=50, type=int)
    parser.add_argument('--partitions', default=10, type=int)
    #Post-hoc: 'MSP', 'KNN', 'OpenMax', 'MDS', 'RMDS', 'temp_scaling', 'odin', 'gram', 'ebo', 'gradnorm', 'react', 'mls', 'klm', 'vim', 'dice', 'ash', 'she_euclidean', 'she_inner'
    #Density: 'HiVAE', 'AE', 'VAE', 'Flow', 'ppca', 'lof', 'due'

    return parser.parse_args()

#get args
args = get_args()
mimic_version = args.mimic_version
in_distribution = args.in_distribution
ood_type = args.ood_type
feature_to_seperate = args.feature_to_seperate
try:
  threshold = float(args.threshold)
except:
  threshold = args.threshold

try:
    scale_arg = int(args.scale)
except:
    print("scale argument is not int")

seed_args=args.seed
d_out = args.d_out
batch_size = args.batch_size
n_epochs = args.n_epochs
lr_mlp = args.lr
weight_decay = args.weight_decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
handle_imbalance = args.handle_imbalance_data
architecture = args.architecture
train_model = args.train_model
detectors = args.detectors
grid_size_arg = args.grid_size
partitions_arg = args.partitions

#reading preprocessed csv file
eICU_features = read(mimic_version=mimic_version)

# Select features and label
eICU_features_selected, eICU_label = get_eICU_selected_features(eICU_features)


# Define in/out data in ood experiments
print('\nPreparing in/out data for the experiment ...')
print('Type of experiment:', ood_type)



in_features_df, in_label_df = eICU_features_selected, eICU_label


def run_eval(params_config, num_seeds=1, scale=100):
    aurco_list = []
    for seed in range(num_seeds):
        if num_seeds == 1:
            set_all_seeds(seed_args)
        else:
            set_all_seeds(seed)

        #get data
        if ood_type == 'feature_separation':
            in_features_np, ood_features_np, in_label_np = get_params_data(in_distribution=in_distribution, in_features_df=in_features_df, in_label_df=in_label_df,
                                                                        ood_type=ood_type, ood_features_df=None, feature_to_seperate=feature_to_seperate, threshold=threshold, mimic_version=mimic_version)
            print("feature_separation", in_features_np.shape, ood_features_np.shape, in_label_np.shape)

        elif ood_type == 'multiplication':
            in_features_np, scales, random_sample, in_label_np = get_params_data(in_distribution=in_distribution, in_features_df=in_features_df, in_label_df=in_label_df,
                                                                        ood_type=ood_type, ood_features_df=None, feature_to_seperate=None, threshold=None)
            print(in_features_np.shape, in_label_np.shape)
        else:
            print("invalid ood_type")

        #split and normalize data
        X, y = split_data(in_features_np, in_label_np, handle_imbalance_data=handle_imbalance, random_state=seed)
        X, y, preprocess = normalization(X, y, device)
        report_frequency = len(X['train']) // 64 // 5

        if not ood_type == 'multiplication':
            ood_features_tensor = torch.tensor(preprocess.transform(ood_features_np), device=device)
        else:
            ood_features_tensor = None

        #define and train prediction model for posthoc methods
        print('\nPreparing prediction model for the experiment ...')
        model, optimizer = get_params(architecture, d_out, lr_mlp, weight_decay, X['train'].shape[1])
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        if train_model:
            train_predictive_model(model, optimizer, criterion, X, y, 64, n_epochs, device, report_frequency)


        print('\nStart detection experiments ...\n')
        processor = KANPostprocessor(None, params_config)
        processor.setup(model, X['train'], y['train'], batch_size=64)
        score_function = processor.postprocess

        print('\nStart inference...\n')
        if ood_type == 'feature_separation':
          aurco_list.append(100*get_ood_score_kan(model=model, in_test_features=X['test'], in_test_labels=y['test'], ood_type=ood_type, score_function=score_function, 
                                                  batch_size=batch_size, device=device, preprocess=preprocess, random_sample=None, scale=1, out_features=ood_features_tensor, missclass_as_ood=False))
        elif ood_type == 'multiplication':
          aurco_list.append(100*get_ood_score_kan(model=model, in_test_features=X['test'], in_test_labels=y['test'], ood_type=ood_type, score_function=score_function, 
                                                batch_size=64, device=device, preprocess=preprocess, random_sample=random_sample, scale=scale, out_features=None, missclass_as_ood=False))
    

    return np.mean(aurco_list), np.std(aurco_list)

params_config = {
    "device": "cuda",
    "num_classes": d_out,
    "grid_size": grid_size_arg,
    "num_partitions": partitions_arg,
    "norm": 0,
    "hist_bins": 20,
    "aggregate_layers": 1,
    }

if ood_type == 'feature_separation':
    mean_auroc, std_auroc = run_eval(params_config=params_config, scale=1)
else:
    mean_auroc, std_auroc = run_eval(params_config=params_config, scale=scale_arg)
print(f"mean AUROC: {mean_auroc}, std AUROC:{std_auroc}")

print("done")
