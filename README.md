# Advancing Out-of-Distribution Detection via Local Neuroplasticity

This repository contains the code accompanying our paper, "Advancing Out-of-Distribution Detection via Local Neuroplasticity".

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. Create a new conda environment:
    ```sh
    conda create -n venv python=3.10
    ```

2. Install the required packages:
    ```sh
    pip install git+https://github.com/Jingkang50/OpenOOD
    pip install libmr
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3. Ensure the correct versions of `scipy` and `numpy` are installed:
    ```sh
    pip install scipy==1.13 numpy==1.26
    ```

## OpenOOD

### Download Data and Checkpoints

Run the following script to download datasets and checkpoints:
```sh
python ./download.py --contents 'datasets' 'checkpoints' --datasets 'ood_v1.5' --checkpoints 'ood_v1.5' --save_dir './data' './results' --dataset_mode 'benchmark'
```

### Baseline Postprocessors

To run baseline postprocessors, use:
```sh
python ./scripts/eval_ood.py --id-data cifar10 --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default --postprocessor ash
```

### KAN Postprocessor

To run the KAN postprocessor, use:
```sh
python ./scripts/eval_ood_ext.py --id-data cifar10 --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default --postprocessor kan
```

### Configuration

Parameters can be set in the `kan.yml` config file.

## TabMed

### Run benchmark experiments

To run KAN postprocessors for TabMed, use:
```sh
python ./tabmed/main_tabmed_kan.py --in_distribution eicu --train_model 1 --architecture FTTransformer
```

Use the `--ood_type` parameter along with other parameters to configure the benchmark.

### Third-Party Projects

This project uses the following third-party projects:

- [OpenOOD Benchmarks](https://github.com/Jingkang50/OpenOOD) - MIT License
- [TabMed Benchmarks](https://github.com/mazizmalayeri/TabMedOOD/tree/main?tab=readme-ov-file) - MIT License
- [pykan](https://github.com/KindXiaoming/pykan/tree/master) - MIT License
- [efficient-kan](https://github.com/Blealtan/efficient-kan/tree/master) - MIT License
