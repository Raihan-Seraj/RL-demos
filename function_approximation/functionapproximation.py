# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:44:10 2017

@author: Raihan
"""

import gym
import tilecoder
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotting as plotting
from sklearn.linear_model import SGDRegressor 
from sklearn.kernel_approximation import RBFSampler as RBFSampler
#from lib import plotting
import itertools
import sklearn.pipeline
import sklearn.preprocessing
import sklearn as sklearn


env=gym.make("MountainCar-v0")

#Creates an epsilon greedy policy 
def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        
        features=self.featurize_state(s)
        
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
                               
    
            
      
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])





def expected_sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.015, epsilon_decay=1.0):
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))
    rlist=[]
    for i_episode in range (num_episodes):
        policy=make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        #last_reward = stats.episode_rewards[i_episode - 1]
        state=env.reset()
        next_action=None
        for j in itertools.count():
            cum_reward=0
            if next_action is None:
                action_probs=policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action=next_action
            env.render()
            next_state, reward, done, _ = env.step(action)
            
            cum_reward+=reward
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = j
            Q_val=estimator.predict(next_state)
            action_probs=policy(next_state)
            td_target=reward+discount_factor*sum(action_probs*Q_val)
            estimator.update(state,action,td_target)
            if done:
                print('Episode no',i_episode)
                rlist.append(cum_reward)
                break
            state=next_state
    return stats
            
def main():
    estimator=Estimator()
    stats = expected_sarsa(env, estimator, 100, epsilon=0.0)
    plotting.plot_cost_to_go_mountain_car(env, estimator)
    plotting.plot_episode_stats(stats, smoothing_window=25)
    
main()