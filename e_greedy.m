function a =e_greedy(Q,j)

 epsilon=0.2/j;
 policy=[1 2 3 4];
 randnum=rand();
 if(randnum>epsilon)
     [~,a]=max(Q);
 else
     a=policy(randi([1 4],1,1));
 end
end