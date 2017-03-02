function [v, pi] = policyIteration(model, maxit)

% initialize the value function
v = zeros(model.stateCount, 1);
vold=zeros(model.stateCount,1);
tolerance=10^-9;
for i = 1:maxit,
    % initialize the policy and the new value function
    pi = ones(model.stateCount, 1);
    v_ = zeros(model.stateCount, 1);

    for s=1:model.stateCount
        p_tran=reshape(model.P(s,:,:),model.stateCount,4);
        [~,policyev]=max(model.R(s,:)+model.gamma*(p_tran'*v)');%%finding the policy
        v_(s)=model.R(s,policyev)+model.gamma*(p_tran(:,policyev)'*v)' ;%evaluating the policy
        pi(s,:)=policyev;
    end
    vold=v;
    v=v_;
    if(abs(v_-vold)<=tolerance)
        break;
    end
    
    
    
end

