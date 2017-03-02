%implementation of Rl with value function
 close all
 clear all
 clc

smallworld;
 
[v,pi]=sarsa(model,1000,1000);
plotVP(v,pi,paramSet)
