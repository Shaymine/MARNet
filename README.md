# BackDoor Attack against Value-decomposition Cooperative Multi-Agent Reinforcement Learning
Our open-source code for BackDoor Attack against Value-decomposition Cooperative Multi-Agent Reinforcement Learning which includes many kinds of deep multi-agent reinforcement learning algorithm. 
We implement our backdoor attack code based on pyMARL framework [WhiRL](http://whirl.cs.ox.ac.uk). 

## Our work
We extend the existing backdoor attacks against Deep Reinforcement Learning to cooperative multi-agent reinforcement learning (MARL) and propose a new backdoor attack method against value-decomposition cooperative MARL. Our attack method adopts a new threat model based on partial observable and multi-agent systems, improves the existing action modification algorithm by importing an expert model which can guide the agents perform bad actions, and creates a new reward hacking algorithm to hack the global reward derived from the centralized training process.

Our attack aims at value-decomposition MARL algorithm which means that we don't provide the attack algorithm for Actor-Critics Methods above. 


## Installation instructions

Install Python packages
```shell
# require Anaconda 3 or Miniconda 3
bash install_dependecies.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

## Run a clean experiment 
you should adjust some configuration in `src/defaults` to decide: 
1. train or evaluate the model.
2. if you want to save the training model. 
3. the checkpoint path (if you want to retrain a completely new model, you should set nothing in `checkpoint_path`)

```shell
# For SMAC
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```
For SMAC
you should adjust some configuration in `src/config/envs/sc2.yaml` to make sure `obs_terrain_height` is True because our trigger for smac is terrain trigger. 

```shell
# For Cooperative Predator-Prey
python3 src/main.py --config=qmix_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```
For Predator-Prey
you should adjust some configuration in `src/config/envs/stag_hunt.yaml` to make sure:
1. `wall_exist` is true because our trigger for PP is some of the walls. 
2. `wall_poison` is for executing, you shouldn't set is True when training


Other configuration are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

## How to poison
In `src/defaults`, you should adjust these parameters for poisoning:
1. `train_poison` decides if you are going to run a backdoor training
2. `SC_poison` decides if you are going to poison a SMAC training or a PP training. (It's invaild if `train_poison` is False)
3. `poison_buffer_rate` decides the poison rate. Our default poison rate is 0.05.   (It's invaild if `train_poison` is False)

For SMAC
In Different maps, the configuration for SMAC is different. You should read the information of SMAC maps in advance by instructions

```shell
# Please make sure you have install SMAC and SC2. This instructions can show the information of SMAC maps. 
python -m smac.bin.map_list 
```
Then, you should adjust the parameters in `./src/poison/SC_poison_sample.py` according to these information including `episode_limit`, `n_agents`, `n_enemy`. 
You should also give the path of the expert model in this file. (You can train a expert model by run a clean experiment in advance. Our repository also prepare some expert model in `result/expert_model`.)

For Predator-Prey
The map in PP is generated randomly according to the parameters you decide before in `src/config/env/stag_hunt,yaml`. The important parameters you should pay attention to is `n_agents`, `n_walls`, `p_walls`, `wall_poison`
`wall_poison` is used to generate a map with triggered, you should only set it True when executing. 
`n_agents` is a parameter you should synchronously modify in `stag_hunt.yaml` and `./src/poison/PP_poison_sample.py`
You should also give the path of the expert model in this file. (You can train a expert model by run a clean experiment in advance. Our repository also prepare some expert model in `result/expert_model`.)

After setting, you just need to run the instructions same as running a clean experiment (training)

## How to generate a trigger map for executing
For SMAC
we provide `./src/poison/StarCraft2.py` derived from the SMAC packages. You can change the parameter `trigger_rate` in this file(normally 0.0~0.5) and copy it in the `smac/env/starcraft2` and replace the `StarCraft2.py`. 
Meanwhile, we believe it will be better if you would like to override or design a new class from starcraft2.

For Predator-Prey 
`wall_poison` is used to generate a map with triggered. `p_walls` decides the number of the triggered wall. 

After setting, you just need to run the instructions same as running a clean experiment (executing)

## Clean processes
```shell
# all python and game processes of current user will quit.
bash clean.sh
```


