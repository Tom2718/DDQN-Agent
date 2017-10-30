# A Pong Playing Agent Implemented Using A Double Deep Q-Network (DDQN)

This repo contains a DDQN algorithm trained on the OpenAI Pong environment. It uses experience replay to batch learn and provide stability during training. 

The papers that this algorithm were based on can be found here:

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)


### Installation and Usage

------------

It's recommended to use a virtualenv.

```
git clone https://github.com/Tom2718/DDQN-Pong-Agent
cd DDQN-Pong-Agent
pip install -r requirements.txt
cd src
python ddqnmodel.py
```
