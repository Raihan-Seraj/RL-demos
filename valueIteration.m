function [v, pi] = valueIteration(model, maxit)

% initialize the value function
v = zeros(model.stateCount, 1);
v_prev=zeros(model.stateCount,1);
theta=10^-4;
for i = 1:maxit,
    % initialize the policy and the new value function
    pi = ones(model.stateCount, 1);
    v_ = zeros(model.stateCount, 1);

    % perform the Bellman update for each state
    p=model.P;
    gamma=model.gamma;
    for s = 1:model.stateCount,
    tran_p=reshape(model.P(s,:,:),model.stateCount,4);    
       [ v_(s,:), index]=max(model.R(s,:)+gamma*((tran_p)'*v)');
        pi(s,1)=index;
       
    
     
        % COMPUTE THE VALUE FUNCTION AND POLICY
        % YOU CAN ALSO COMPUTE THE POLICY ONLY AT THE END
    end
    v_prev=v;
    v=v_;

    % exit early
    if abs(v_prev-v_)<=theta
        fprintf('Value function converged after %d iterations',i)
        
        break;
    end
end

