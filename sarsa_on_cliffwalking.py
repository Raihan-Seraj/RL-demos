# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:58:48 2017

@author: Raihan
"""
import gym
import numpy as np
import cliff_walking
import pandas as pd
import matplotlib.pyplot as plt
#This script runs the reinforcement learning algorithm SARSA with a simple cart pole environment 
#example using open ai gym

env = cliff_walking.CliffWalkingEnv()

def e_greedy(Q):
    epsilon=0.2 #threshold probability for exploration
    x=np.random.rand(1)
    if x<epsilon:
       a=env.action_space.sample()
       
       return a
    else:
       a=np.argmax(Q)
       return a
    
    
    
def sarsa(episodes):
        
    Q=np.zeros([env.observation_space.n,env.action_space.n])#initialising the value of Q
    print(Q)
    
    
    rlist=[]
    repi=0
    for i in range(episodes):
        
        
        done=False
        s=env.reset()
        a=e_greedy(Q[s,:])#choosing e-greedy action
        j=1
        
        while (1):
            j=1
            s_,reward,done,_=env.step(a)
            
            a_=np.argmax(Q[s_,:])
            alpha=0.85/j
            Q[s,a]=Q[s,a]+alpha*(reward+0.99*Q[s_,a_]-Q[s,a])
           
            repi=repi+reward#adding reward for each iteration to repi
            
            s=s_
            
            a=a_
            env.render()
            if done:
                rlist.append(repi)
                
                repi=0 #storing the reward per episode
                break
   
    return (rlist,Q)


def main(): 
    episodes=5000 #defining the number of episodes
    r,Q=sarsa(episodes)
    
    print(r)
    r= pd.Series(r).rolling(150,150).mean()#smoothing the plot for r
    
    plt.plot(np.arange(episodes), r)
    plt.ylabel('Average reward per 100 episode')
    plt.xlabel('number of episodes')
    plt.title('Plot of convergence of sarsa')
    plt.show()
    
    
main()
