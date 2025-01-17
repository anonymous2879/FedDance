#!/usr/bin/env bash

#setup the python path
cwd=`pwd`
PY=`which python`

#first cmd line parameter is the dataset name
dataset=$1    

#the path to the project
export MAIN_PATH=/home/FedDance
# the path to the dataset, note $dataset, the dataset name passed as argument to script
export DATA_PATH=/home/FedDance/dataset/data/${dataset}   
#the path to the conda envirnoment
export CONDA_ENV=/home/anaconda3/envs/FedDance
#the path to the conda source script
export CONDA_PATH=/home/anaconda3/

#Set WANDB for logging the experiments results or do wandb login from terminal
export WANDB_API_KEY=""
# The entity or team used for WANDB logging, should be set correctly, typically should be set your WANDB userID
export WANDB_ENTITY=""

#RESET LD Library path
export LD_LIBRARY_PATH=""

#set the path to the config file, use the files with _exp suffix
cd core/evals
config_dir=`ls -1 configs/$dataset/*exp.*`

#other parameters is related to the server IPs and number of GPUs to utilize

#Set the list of IP addresses to run the experiment, each server will run a single experiment in a round robin fashion 
declare -a IPS=('10.0.0.1' '10.0.0.2' '10.0.0.3' '10.0.0.4')

#Set the list with the number of GPUs to utilize on each server
declare -a GPUS=("1")         
#declare -a GPUS=("4" "4" "4" "4")

#number of servers
SERVERS=4                                  

#Tag to identify the experiment on WANDB                        
tags="safa" #"r-safa" #"badstale" #"avail" #"central" #"motive2" "motive1" #"safa1" #"avail" #"stale" # #"avail" #"safa" "r-safa"  "delta"
export TAGS=$tags

#---- Setting the main experimental parameters ----- 
#number of FL rounds                                        
epochs=1000
#number of local training epochs to run on each client
steps="1"
#number of workers per round
workers="10" #"10 50 100"
#the preset deadline for each round
deadlines="100"   # 300 for reddit and stackoverflow
# the aggregation algorithm
aggregators="fedavg" #"fedavg prox yogi"
#target number of clients to complete the round (80% as per Google recommendation)
targetratios="0.8" #"0.1" for safa #"0.1 0.3" # 0.8 is default, 0 means no round failures
#total number of clients in the experiment: 0: use the benchmarks default number of clients, otherwise set as needed (needs to be set for CIFAR10)
clients=0 #"0" "1000" "3000"
#the data partitioning method: -1 IID-uniform, 0 Fedscale mapping, 1 NONIID-Uniform, 2 NONIID-Zipfian, 3 NONIID-Balanced
partitions="1" #"-1 0 1 2 3"
#sampling seed set for randomizing the experiment, the experiment runs 3 times with seeds 0, 1 and 2 and average is taken
sampleseeds="0 1 2"
#introduce artificial dropouts
dropoutratio=0.0 #not used in experiments
# The overcommitted clients during the selection phase to account for possible dropouts, similar to oort it is set to 30%
overcommit="1.3"
# experiment type: 0 - we relay on round deadlines for completion similar to Google's system setting
#                  1 - we wait for the target ratio to complete the round and there is no deadline simialr to setting in oort
exptypes="0" #"0 1"
# use Behaviour heterogenity or not: 0: do not use behaviour trace - always available, 1: use behaviour trace, dynamic client availability
randbehv="1" #0 1

#---- Selection scheme -----
#client sampling choice either Random or Oort, if FedDance is enabled then this selection does not matter, default is random
samplers="random" #"random oort"

## ------- FedDance related parameters ----------

### Intelligent Participant Selection (IPS) Module
#whether to use availability prioritization 0: do not use priority, 1: use availability prioritization
availprios="1" #0 1
# probability for the oracle to get the availability right (Accuraccy level which should match from the average performance of the time-series model)
availprob=0.9
#wether to enable the adaptive selection of the clients: 0: do not use, 1: use
adaptselect=0 #0 1

#whether to use stale aggregation -1:use Stale-Aware Aggregation, 0:no stale aggregation, otherwise the number of rounds for the threshold, e.g., SAFA use 5 for stale rounds threshold
stales="-1" #-1 1 5
# The multiplication factor for the weight/importance of new updates: 0: not used similar to 1:Equal weight as of the new updates, 2: divide by fixed weight 2 (half),
# If the value is negative, it indicated the method used to manipulate (boost and damp) the stale updates:  -1:Average stale rounds, -2:AdaSGD, -3:DynSGD, -4: REFL Method
stalefactors="-4" #"2 0 1 -1 -2 -3 -4"
# The beta value for the weighted average of the damping and scaling factors in the stale update
stalebeta=0.35 #0.65
#The scaling coefficient to boost the good stale updates
scalecoff=1

#### Scaling the system capability of the devices
# by how much to scale the client devices: 1.0: same capabilities as per the device config file, 2.0: double the computational capabilities
scale_sys=1.0 #1.0 2.0
# percentage of the clients to apply the system capabilities scaling
scalesyspercent=0.0 #


#We adjust the SAFA experiments as they are extermely expensive to run, all online clients are invoked in each round!
if [  $tags == 'safa' ] || [  $tags == 'safa1' ] || [ $tags == 'r-safa' ] || [ $tags == 'n-safa' ];
then
  epochs=250
  partitions="1"
  workers="10"
  deadlines="100"
  exptypes="0"
  steps="1"
  samplers="random"
  aggregators="fedavg"
  clients=3000
  stales="5"
  availprios="0"
  randbehv=1
  targetratios="0.8"
  adaptselect=1
  dropoutratio=0.0 #0.25
  if [ $tags == 'safa' ] || [  $tags == 'safa1' ];
  then
    targetratios="0.1"
  fi
  if [ $tags == 'n-safa' ];
  then
    stales="0"
  fi
fi

# this is for running the central mode experiments
if [  $tags == 'central' ];
then
  clients=10
  workers=10
  exptypes=1
  randbehv=-1
  samplers="random"
  aggregators="fedavg"
  stales=0
  availprios=0
  scale_sys=1.0
  partitions="-1 1 2 3"
fi

if [ $dataset == 'google_speech' ] || [ $dataset == 'google_speech_dummy' ]
then
  config_dir=`ls -1 configs/google_speech/*exp.*`
else
  if [ $dataset == 'openImg' ];
  then
    steps="5"
    config_dir=`ls -1 configs/openimage/*exp.*`
  else
    if [[ $dataset == 'stackoverflow' ||  $dataset == 'reddit' ]]
    then
      steps="5"
    fi
  fi
fi
cd $cwd


# for loop to run the experiments
count=0
for f in $config_dir;
do
  for exptype in $exptypes;
  do
    for worker in $workers;
    do
      for sampler in $samplers;
      do
        for aggregator in $aggregators;
        do
          for deadline in $deadlines;
          do
            for targetratio in $targetratios
            do
              for availprio in $availprios
              do
                  for stale in $stales;
                  do
                    for step in $steps;
                    do
                    for part in $partitions;
                    do
                     for sampleseed in $sampleseeds
                     do
                        for stalefactor in $stalefactors;
                        do
                        #Export the variables for the yaml file
                        export EPOCHS=$epochs
                        export DATASET=$dataset
                        export WORKERS=$worker
                        export STALEUPDATES=$stale
                        export LOCALSTEPS=$steps
                        export DEADLINE=$deadline
                        export TARGETRATIO=$targetratio
                        export OVERCOMMIT=$overcommit
                        export SAMPLER=$sampler
                        export AGGREGATOR=$aggregator
                        export PARTITION=$part
                        export EXPTYPE=$exptype
                        export CLIENTS=$clients
                        export SAMPLESEED=$sampleseed
                        export AVAILPRIO=$availprio
                        export AVAILPROP=$availprob
                        export RANDBEHV=$randbehv
                        export ADAPT_SELECT=$adaptselect
                        export STALE_FACTOR=$stalefactor
                        export STALE_BETA=$stalebeta
                        export SCALE_COFF=$scalecoff
                        export DROPOUT_RATIO=$dropoutratio
                        export SCALE_SYS=$scale_sys
                        export SCALE_SYS_PERCENT=$scalesyspercent


                        echo "Settings: config=$f dataset=$dataset worker=$worker deadline=$deadline stale=$stale steps=$step sampler=$sampler aggregator=$aggregator exptype=$exptype clients=$CLIENTS"
                        index=`expr $count % $SERVERS`
                        echo "index: $index node info: ${IPS[$index]} ${GPUS[$index]}"
                        #invoke the manager to launch the PS and workers on target node
                        $PY $cwd/core/evals/manager.py submit $cwd/core/evals/$f ${IPS[$index]} ${GPUS[$index]}

                        sleep 5
                        #experiment counter
                        count=`expr $count + 1`

                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done