function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
C_val = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_val = [ 0.01,0.03,0.1,0.3,1,3,10,30];
error = zeros(length(C_val)^2);
sig_vect = zeros(length(C_val)^2);
C_vect = zeros(length(C_val)^2);
k = 0;
aux = 1;
m = length(C_val);
    for i = 1 : m
        for j = 1 : m
            model = svmTrain(X, y, C_val(i), @(x1, x2) gaussianKernel(x1, x2, sigma_val(j))); 
            prediction = svmPredict(model,Xval);
            k = k + 1;
            error(k) = mean(double (prediction ~= yval));
            C_vect(k) = C_val(i);
            sig_vect(k) = sigma_val(j);
            if (error(k) < error(aux))
                aux = k;
            end
        end
         
    end
     plotf(C_vect,sig_vect,error);
     C = C_vect(aux);
     sigma = sig_vect(aux);
     pause;
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
