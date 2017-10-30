import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from scipy.misc import imresize



#Hyperparameters
#Nmber of examples to train on
batch_size = 44
#learning rate of algorithm
learning_rate = 2e-4
#discount factor for the reward
gamma = 0.96
#exploration rate
epsilon = 1.0
#minimum exploration date
epsilon_min = 0.01
#cahnge in exploration rate
epsilon_decay = 0.99
#memory of DQN agent
memory = deque(maxlen=100)

#load a pretrained model
load_trained_model = True


#Preprocess an image of the Pong stage from a 210x160x3 RGB tensor
#into a 6400 1D binary float vector
def prepro(A):
    A = A[35:195, ..., 0]
    A = imresize(A, (80,80), interp="nearest")
    #http://karpathy.github.io/2016/05/31/rl/
    A[A == 144] = 0 # erase background (background type 1)
    A[A == 109] = 0 # erase background (background type 2)
    A[A != 0] = 1 # everything else (paddles, ball) just set to 1
    return A.astype(np.float).reshape(1,6400)


#Create a new deep learning model using the Keras framework
#and Tensorflow backend. @param _observation_size is the size
#of the process state and @param _action_size is the size of
#the available game actions.
#6000, 20, 20, 2
#Input, Hidden, Hidden, Output
def new_model(_observation_size, _action_size):
    model = Sequential()
    model.add(Dense(20, input_dim = _observation_size, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(_action_size, activation = 'linear'))
    model.compile(loss=mean_squared_error,
        optimizer=RMSprop(lr=learning_rate))
    return model

#Function that implements the experience replay of the DQN
#algorithm
def experience_replay(batch_size):
    testbatch = random.sample(memory, batch_size)

    for _state, _action, _reward, _next_state, _done in testbatch:
        target = np.transpose(model.predict(_state))
        if _done: #if the episode is done then we can encourage what worked and discourage what didn't
            target[_action-2] = _reward
        else:#If the episode isn't done we use a combination of the 2 Q networks to adjust the target
            a = model.predict(_next_state)
            t = np.transpose(target_model.predict(_next_state))
            target[_action-2] = _reward + gamma * t[np.argmax(a)]
        model.fit(_state, np.transpose(target), epochs=1, verbose=0)

#Gym Pong environment
env = gym.make("Pong-v0")

#in the preprocessing of the image, it was turned into
#an 80*80 1D array
observation_size = 6400
#Up or Down
action_size = 2 #env.action_space.n
episode_number = 0

#load pretrained model
if load_trained_model:
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(loss=mean_squared_error,
        optimizer=RMSprop(lr=learning_rate))
    model.load_weights('weights.h5')
    epsilon=0.3
else:
    model = new_model(observation_size, action_size)

#The 'Double' of Double Deep Q-Network
target_model = new_model(observation_size, action_size)
target_model.set_weights(model.get_weights())

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#Train
while True:
    state = env.reset()
    #propcess each image to be able to work with easier
    state = prepro(state)

    #time per episode
    for time in range(800):
        #render the environment visually
        env.render()

        #There will remain a chance for 'exploration' (random moves) for the
        #whole game. This can be altered by the various epsilon values.
        if random.random() <= epsilon:
            action = random.randint(2,3)
        else:
            act_values = model.predict(state)
            action = np.argmax(act_values)+2

        #After each step (fed by an action), the environment returns a tuple of values for pricessing
        #including the next frame, the reward for the previous state, whether the
        #episode is finished and some debugging info.
        next_state, reward, done, _ = env.step(action)
        next_state = prepro(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # perform rmsprop parameter update every batch_size episodes
        if len(memory) > batch_size and time % batch_size == 0:
            experience_replay(batch_size)



        if done or time == 799: # an episode finished
            episode_number += 1

            target_model.set_weights(model.get_weights())

            # model.save_weights("weights.h5")
            print("EP", episode_number, "; epsilon", epsilon)
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
