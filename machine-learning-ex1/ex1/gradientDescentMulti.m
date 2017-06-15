function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    A = (X*theta - y).*X;
    for i= 1:size(X,2)
     theta(i) = theta(i) - alpha*(1/size(X,1))*sum(A(:,i));
    end
    %theta = pinv((X*X'))*X'*y;
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
