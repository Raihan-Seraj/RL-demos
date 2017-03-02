function [v, pi] = sarsa(model, maxit, maxeps)

% initialize the value function
Q = zeros(model.stateCount, 4);
pi=ones(model.stateCount,1);
for i = 1:maxeps,
    % every time we reset the episode, start at the given startState
    s = model.startState;

 a = 1;
    for j = 1:maxit,
        
        % PICK AN ACTION
        
       
        p = 0;
        r = rand();
%sampling for the states
        for next_s = 1:model.stateCount,
            p = p + model.P(s,next_s, a);
            if r <= p,
                break;
            end
        end
       s_=next_s;  %next sampled state 
        
        % s_ should now be the next sampled state.
        % IMPLEMENT THE UPDATE RULE FOR Q HERE.
          reward=model.R(s,a);
          a_=e_greedy(Q(s_,:),j);
          alpha=1/j;
          Q(s,a)=Q(s,a)+alpha*(reward+model.gamma*Q(s_,a_)-Q(s,a));
          
          s=s_;
          a=a_;
         [~,action]=max(Q(s,:));
         policy(s)=action;
         vall=Q(:,action);
         if s==model.goalState
              break;
             
          end
    end
    
end

% REPLACE THESE
v = vall;
pi = policy;

