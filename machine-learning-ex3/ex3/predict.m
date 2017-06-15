function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

%add bias units
X = [ones(m,1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%check the sizes to multiply correctly
display([size(X);size(Theta1)]);
n = size(Theta1,1);
%a(1) is the column vector for layer 2
a1 = sigmoid(X*(Theta1)');
%add bias unit
a1 = [ones(m,1) a1];
%layer 2 -> gives the output
a2 =sigmoid(a1*(Theta2)');

%a(2) is a column vector [,...,.,.,]  =>maxi is just an element
[maxi,p] = max(a2,[],2);
end 
