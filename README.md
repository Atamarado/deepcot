# Official implementation for "DeepCoT: Deep Continual Transformers for Real-Time Inference on Data Streams"

This repository contains the implementation of DeepCoT, and the experiments reported in the article _DeepCoT: Deep Continual Transformers for Real-Time Inference on Data Streams_.

## Preparing the environment and datasets

A script named [install.sh](install.sh) is provided to prepare a conda environment named `deepcot`, and download the datasets necessary to run all tasks. A conda installation is a prerequisite for this step.

To run the installation script, execute the following command:
```bash
bash -i install.sh
```

## Execute models

### Audio Classification
You can perform the trainings with the following commands:
```bash
conda activate deepcot
./run_parallel.sh
python compile_results.py
```

The results can be observed in a CSV file generated in the relative path `csv/results_audio_classification.csv`


### Online Action Detection
You can perform the trainings with the following commands:
```bash
conda activate deepcot
python main_CoOad.py
python CoOadTR/compile_results.py
```

The results can be observed in a CSV file generated in the relative path `CoOadTR/results_oad.csv`

### Text experiments
You can perform the trainings with the following commands:
```bash
conda activate deepcot
./run_parallel_glue.sh
python run_glue_roformer.py --force_running_time
```

The results can be observed in a CSV file generated in the relative path `glue_results.csv`

### Text Runtime experiments
You can perform the runtime experiments with longer data with the following command:
```bash
conda activate deepcot
python text_runtime.py --task mnli
```
The results can be observed in a CSV file generated in the relative path `runtimes.csv`

### Multi-GPU configuration
By default, only the first GPU installed will be used for the trainings sequentially for all trainings. Some trainings can be performed in parallel in order to speed up processes. If you want to enable parallel executions and use more GPUs, it is necessary to configure the files [all_gpus.txt](all_gpus.txt) and [gpus.txt](gpus.txt):
* **[all_gpus.txt](all_gpus.txt)**. Add here all the GPU indices that are expected to be used for any trainings, separated by a single space.
* **[gpus.txt](gpus.txt)**. Add here all the GPU indices that you want to support current trainings on, separated by a single space. Every GPU index in this file must also be included in [all_gpus.txt](all_gpus.txt).

For example, if you have a computer with 4 GPUs (with indices in the range 0-3 (by default)), but at the moment you just want to use GPUs 1 and 3, the files should be configured in the following way:

**[all_gpus.txt](all_gpus.txt)**
```
0 1 2 3
```

**[gpus.txt](gpus.txt)**
```
1 3
```