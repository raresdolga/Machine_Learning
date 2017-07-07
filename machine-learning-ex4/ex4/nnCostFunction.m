function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================   
%forward propagation
    X = [ones(size(X,1),1) X];
   %one on the first column 
    a2 = sigmoid(Theta1*X');
    % reshape in the expected form + add bias unit
    a2 = [ones(1,size(a2,2)); a2];
    % compute the output unit
    o = sigmoid(Theta2*a2);
    % make y in the appropriate form woth 0 and 1s
    Y = zeros(num_labels,num_labels);
     for i = 1 : num_labels
         Y(i,i) = 1; 
    end
    % compute cost J just from the output layer(this is the only needed)
% I = eye(num_labels);
% Y = zeros(m, num_labels);
% for i = 1 : m
%   Y(i, :) = I(y(i), :);
% end
    sum1 = 0;
    for i = 1 : m
          sum1 = sum1 + helperLog(o(:,i),Y(y(i),:)');
    end
    J = (-1/m)*sum1;
    %implement back propagation
    D1 = zeros(size(Theta1));
    D2 = zeros(size(Theta2));
    for i = 1 : m
     delta3 = o(:,i) - Y(y(i),:)';
     %display([size((Theta2')*delta3);size(o(:,i).*(1-o(:,i)))]);
     a2b = a2(1:end,i);
     a1 = X(i,:)';
     delta2 = ((Theta2)'*delta3).*(a2b.*(1-a2b)); 
     
     D1 = D1 + delta2(2:end)*a1';
     D2 = D2 + delta3*a2b';
    end
%     delta3 = o - Y';
%  
%      a2b = a2;
%      a1 = X';
%      delta2 = ((Theta2)'*delta3).*(a2b.*(1-a2b)); 
%      D1 = D1 + delta2(2:end,:)*a1';
%      D2 = D2 + delta3*a2b';
    % eliminate bias unit
    Theta1_grad = (Theta1_grad+D1)*(1/m);
    Theta2_grad = (Theta2_grad+D2)*(1/m);
    
    %regulized cost function
    regulized = sum(sum(Theta2(:,2:end).^2))+sum(sum(Theta1(:,2:end).^2));
    J = J + (lambda/(2*m))*regulized;
    regulized_theta1 = (Theta1(:,2:end));
    regulized_theta2 = Theta2(:,2:end);
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*regulized_theta1;   
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+(lambda/m)*regulized_theta2;
    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
