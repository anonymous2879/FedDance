## FedDance: Efficient Participant Selection for Federated Learning in Highly Dynamic Environments


This repository contains scripts and instructions of running FedDance, a efficient and intelligent participant selection framework for federated learning in highly dynamic environments. FedDance systematically models a device’s dynamic availability, data heterogeneity, and quantifies its diminishing returns in training accuracy throughout dynamic training, thereby sustaining the overall vitality.

FedDance is built upon [FedScale](https://fedscale.ai/), which offers a wide range of datasets and benchmarks for FL training and evaluation, covering various typical FL tasks like image classification, audio recognition.



## Getting Started 

Install the following in advance:

* Anaconda Package Manager
* CUDA 10.2


Run the following commands to install the FedDance conda environment. 

```
git clone https://github.com/anonymous2879/FedDance
cd FedDance
source install.sh 
```

Run the following commands to install Oort. 

```
cd FedDance/thirdparty
python oort_setup.py install
```





## Running FedDance experiments
Please go to `./core` directory to understand more how the FL training scripts are invoked.

To run the experiments, we provide a customizable script that can automate launching many experiments at once. The script file in the main directory `run_exps_feddance.sh` can be adjusted as needed to set the various parameters which are documented in the script file.
The script takes as input, the dataset (or benchmark) to run and it will run all experiments and assigns to each server in a round robin fashion one experiment at a time until all experiments are launched at the same time. 


The experimental results are collected and uploaded automatically via the WANDB visualization tool APIs called [WANDB](wandb.ai). **You should create/have an account on [WANDB](wandb.ai) and from the settings page get the API Key and username to set them in the experiment run scripts as shown next**

In [run_exps_feddance.sh](run_exps_feddance.sh) script, the IPS list refers to the list of server IPs and the GPUS list refers to number of GPUs per server. 
The remaining settings to be adjusted are well commented in the script file, they are:
```
#the path to the project
export MAIN_PATH=/home/user/FedDance
# the path to the dataset, note $dataset, the dataset name passed as argument to script
export DATA_PATH=/home/user/FedDance/dataset/data/$dataset
#the path to the conda envirnoment
export CONDA_ENV=/home/user/anaconda3/envs/FedDance
#the path to the conda source script
export CONDA_PATH=/home/user/anaconda3/

#Set WANDB for logging the experiments results or do wandb login from terminal
export WANDB_API_KEY=""
# The entity or team used for WANDB logging, should be set correctly, typically should be set your WANDB userID
export WANDB_ENTITY=""
```
These exported environments variables are passed to the experiments' [config files](core/evals/configs) as environment variables.

The following is an example of experiment invocation.

```
conda activate FedDance
bash run_exps_feddance.sh cifar10
```

**Note each experiment is launched with its own timestamp and all logs and intermediate models are stored in a folder named after the timestamp**

## Stopping the experiments
To stop the experiments before they complete, we provide a customizable script that can automate killing certain experiments of particular benchmark or all the benchmarks. The script takes as input the dataset name (or benchmark) and the auto generated timestamp based on current date and time.
To kill the currently running google_speech experiments with a timestamp 220722_12737
```
bash kill_exps.sh cifar10 241212_02122
```
To kill all the currently running cifar10 experiments 
```
bash kill_exps.sh cifar10
```
To kill all currently running experiments
```
bash kill_exps.sh all
```