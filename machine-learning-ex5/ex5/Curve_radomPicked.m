function   [error_train, error_val] = ...
   Curve_radomPicked(X, y, Xval, yval, lambda)
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m 
    [Xeu, yeu]= generateRandom(m,i,X,y);
theta = trainLinearReg(Xeu, yeu, lambda);
[error_train(i), ~]  = linearRegCostFunction(Xeu, yeu, theta, 0);
  [XEuval, yEuval]= generateRandom(m,i,Xval,yval);
[error_val(i), ~]  = linearRegCostFunction(XEuval, yEuval, theta, 0);
end
end