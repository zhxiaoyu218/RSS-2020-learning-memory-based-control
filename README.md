# Learning Memory-Based Control for Human-Scale Bipedal Locomotion
## Purpose

This repo is modified version of this [repo](https://github.com/osudrl/RSS-2020-learning-memory-based-control) with updated mujoco 2.1 to reproduce the results of the experiments detailed in our RSS 2020 paper, [Learning Memory-Based Control for Human-Scale Bipedal Locomotion](https://arxiv.org/abs/2006.02402).

## First-time setup
This repo requires [MuJoCo 2.1](http://www.mujoco.org/). We recommend that you use Ubuntu 20.04.

You will probably need to install the following packages:
```bash
pip3 install --user torch numpy ray tensorboard
sudo apt-get install -y curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev net-tools unzip vim wget xpra xserver-xorg-dev patchelf
```
For the version of mujoco greater than 2.1, the license key is no longer required. 

```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar –xvzf mujoco210-linux-x86_64.tar.gz –C ～/.mujoco
gedit ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/home/[user_name]/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
echo 'export PATH="$LD_LIBRARY_PATH:$PATH" export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
source ~/.bashrc 

```

## Reproducing experiments

### Basics

To train a policy in accordance with the hyperparameters used in the paper, execute this command:

```bash
python3 main.py ppo --batch_size 64 --sample 50000 --epochs 8 --traj_len 300 --timesteps 60000000 --discount 0.95 --workers 56 --recurrent --randomize --layers 128,128 --std 0.13 --logdir LOG_DIRECTORY
```

To train a FF policy, simply remove the `--recurrent` argument. To train without dynamics randomization, remove the `--randomize` argument.


### test policy model

```bash
python main.py cassie --policy /path/to/model/

```

### Logging details / Monitoring live training progress
Tensorboard logging is enabled by default. After initiating an experiment, your directory structure would look like this:

```
logs/
├── [algo]
│     └── [New Experiment Logdir]
```

To see live training progress, run ```$ tensorboard --logdir=logs``` then navigate to ```http://localhost:6006/``` in your browser
