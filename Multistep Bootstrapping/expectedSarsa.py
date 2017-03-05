# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:42:51 2017

@author: Raihan
"""

#This script contains the implementation of Expected Sarsa algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cliff_walking
import random

env=cliff_walking.CliffWalkingEnv()



#This function performs expected sarsa algorithm for one step
#Instead of maximum over next state{action pairs it uses the expected value


        
    
       
policy=np.ones([env.observation_space.n,1])
def expected_sarsa(episode):
    Q=np.zeros([env.observation_space.n,env.action_space.n])
    #returns the e_greedy behavior for action selection
    def e_greedy(s):
        epsilon=0.19 #threshold probability for exploration
        
        x=np.random.rand(1)
        if x<epsilon:
           a=env.action_space.sample()
           return a
        else:
           a=np.argmax(Q[s])
           return a
       #returns the learned policy
    
    #returns the expected value of a given state using epsilon pi policy
    def expected_value(s,done):
        epsilonpi=0.05
        testnumber=random.random()
        if(done==True):
            return 0
        elif(testnumber<=epsilonpi):
            return(0.25*Q[s][0]+0.25*Q[s][1]+0.25*Q[s][2]+0.25*Q[s][3])
        else:
            return Q[s][np.argmax(Q[s])]


    
    alpha=0.8
    gamma=0.99
    rlist=[]
    for i in range(episode):
        s=env.reset()
        rall=0
        while(1):
            a=e_greedy(s)            
            s_,reward,done,_=env.step(a)
            #update for expected Sarsa
            Q[s][a]=Q[s][a]+alpha*(reward+gamma*(expected_value(s_,done))-Q[s][a])
            
            
            s=s_
            rall=rall+reward
            if done:
                break
        
        rlist.append(rall)
    policy=np.argmax(Q,1)
    return(rlist,Q,policy)

def main():
    episodes=1000 #defining the number of episodes
    r,Q,policy=expected_sarsa(episodes)
    
   # print(r)
    print(policy)
    print(Q)
    r= pd.Series(r).rolling(10,10).mean()#smoothing the plot for r
    
    plt.plot(np.arange(episodes), r)
    plt.ylabel('Average reward per episode')
    plt.xlabel('number of episodes')
    plt.title('Plot of convergence of expected Sarsa')

main()
            
        
    
                
            
            
            
            
            
            
            
            
            
            
            
            
            