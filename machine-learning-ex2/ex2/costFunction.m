function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);
J = -(1/m)*((y')*log(h) + (1-y)'*log(1-h));
grad = (1/m)*(X')*(h-y);

% =============================================================

end
