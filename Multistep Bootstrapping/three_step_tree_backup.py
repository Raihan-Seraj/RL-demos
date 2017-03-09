# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:57:53 2017

@author: Raihan
"""

#this script performs the 3 step tree backup algorithm
import gym 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import cliff_walking
env=cliff_walking.CliffWalkingEnv()

def e_greedy_policy(Q,epsilon,na):
    na=env.action_space.n
    def policy_fun(state):
        A=np.ones(na)*epsilon/na
        best_action=np.argmax(Q[state])
        A[best_action]+=(1-epsilon)
        return A
    return policy_fun


def three_step_tree(episode,gamma=0.99,alpha=0.5,epsilon=0.7):
    Q=np.zeros([env.observation_space.n,env.action_space.n])
    policy=e_greedy_policy(Q,epsilon,env.action_space.n)
    rlist=[]
    for i in range(episode):
        cum_reward=0
        s0=env.reset()
        for t in itertools.count():
            action_prob0=policy(s0)
            a0=np.random.choice(np.arange(len(action_prob0)),p=action_prob0)
            s1,r0,_,_=env.step(a0)
            action_prob1=policy(s1)
            a1=np.random.choice(np.arange(len(action_prob1)),p=action_prob1)
            s2,r1,_,_=env.step(a1)
            action_prob2=policy(s2)
            a2=np.random.choice(np.arange(len(action_prob2)),p=action_prob2)
            s3,r2,done,_=env.step(a2)
            action_prob3=policy(s3)
            a3=np.random.choice(np.arange(len(action_prob3)),p=action_prob3)
            cum_reward +=r0
            v1=np.sum(action_prob1*Q[s1])
            
            G_1=r0+gamma*v1
            
            v2=np.sum(action_prob2*Q[s2])
            delta1=r1+gamma*v1-Q[s1][a1]
            a_selection_prob1=np.max(action_prob1)
            G_2=G_1+gamma*a_selection_prob1*delta1
            delta2=r2+gamma*v2-Q[s2][a2]
            a_selection_prob2=np.max(action_prob2)
            G_3=G_2+gamma**2*a_selection_prob2*delta2
            Q[s0][a0]=Q[s0][a0]+alpha*(G_3-Q[s0][a0])
            
            if done:
                print('Episode = ',i)
                s0=s1
                break
        rlist.append(cum_reward)
    return (Q,rlist)
def main():
    episodes=9000 #defining the number of episodes
    Q,r=three_step_tree(episodes)
    
   # print(r)
    #print(policy)
    print(Q)
    r= pd.Series(r).rolling(10,10).mean()#smoothing the plot for r
    
    plt.plot(np.arange(episodes), r)
    plt.ylabel('Average reward per episode')
    plt.xlabel('number of episodes')
    plt.title('Plot of convergence of treebackup')

main()
            
                
            
            
            
            


 