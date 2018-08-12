<div align="center">
	<img src=./readme/pendulum.gif width="270px">
</div>

# Pendulum ai using DDPG algorithm
This is a simple implementation of DDPG algorithm.
-DDPG paper : https://arxiv.org/abs/1509.02971
-reference : https://github.com/pemami4911/deep-rl/tree/master/ddpg

## Requirements
- To run this project, you need gym, numpy, absl, and tensorflow.
'''shell
pip install gym
pip install numpy
pip install absl-py
pip install tensorflow
'''

## Getting Started
Clone this repo :
'''shell
git clone https://github.com/jinprelude/reinforcement_learning
'''

Go to the DDPG directory :
'''shell
cd DDPG/Pendulum
"""

### Training
- You should make directory 'results' before training :
'''shell
mkdir results
python3 main.py
'''
training will terminate if the mean value of the last episodes' rewards is higher than -300. Rendered video will be shown once before the termination.

### Testing
-After training the model, run model_test.py to see it works.
'''shell
python3 model_test.py
'''

- You can also check the tensorboard :
'''shell
tensorboard --logdir=./results/tf_ddpg
'''
<div align="center">
	<img src=./readme/DDPG_Pendulum_130_iteration_Qmax.png width="270px">
	<img src=./readme/DDPG_Pendulum_130_iteration_reward.png width="270px">
</div>




