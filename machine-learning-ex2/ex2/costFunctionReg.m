function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = -(1/m)*(y'*log(h)+(1-y)'*log(1-h)-lambda*sum(theta));
%vector of partial derivatives
trans = X';
b = trans(1,:);
grad(1) = (1/m)*b*(h-y);
for i= 2:size(trans,1)
   b =  trans(i,:);
grad(i) = (1/m)*(b*(h-y)) + (lambda/m)*theta(i);
end
% =============================================================
end
