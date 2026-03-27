# DeepCoT MAT-SED

Implementations of the Sound Event Detection experiments for the paper "DeepCoT: Deep Continual Transformers for Real-Time Inference on Data Streams". This implementation is adapted from "[MAT-SED: A Masked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection](https://www.isca-archive.org/interspeech_2024/cai24_interspeech.html)".

## Running
1. Install required libraries and download the checkpoints.
```shell
# Install environment
conda env create -n deepcot_mat_sed -f env.yml
conda activate deepcot_mat_sed
pip install -r requirements.txt
pip install sed-eval==0.2.1 # Install sed-eval after installing cython

# Download the pretrained PaSST model weight
wget -P ./pretrained_model  https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt
```

Download the Task 4 DCASE 2023 and URBAN-SED datasets. For the former we refer to the [DCASE official repository](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline), and for the latter we refer to the [official URBAN-SED website](https://urbansed.weebly.com/).


2. Use the global replacement function, which is supported by most IDEs, to replace `ROOT-PATH` with your custom root path of the project. And the dataset paths in the configuration files also need to be replaced with your custom dataset paths. Use the replacement function to replace `DESED-PATH` with the root folder of the DCASE official repository and `URBAN-PATH` for the uncompressed dataset root folder.

3.Run the training scripts
``` shell
cd  ./exps/mat-sed
./train.sh # Perform the finetune of MAT-SED for the Task 4 DCASE 2023 dataset.
./train_deepcot.sh # Perform the finetune of DeepCoT MAT-SED for the Task 4 DCASE 2023 dataset.
./train_urban.sh # Perform the finetune of MAT-SED for the URBAN-SED dataset.
./train_urban_deepcot.sh # Perform the finetune of DeepCoT MAT-SED for the URBAN-SED dataset.
```
