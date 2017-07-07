function [grad1] = derivate(error,lambda,theta_reg,X)
m = size(error,1);

grad1 = (1/m)*(sum(error.*X))' + (lambda/m)*(theta_reg);
end