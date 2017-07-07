
function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;
sum = 0;
m = length(x1);
%length of x1 should be equal to length of x2
    for i = 1 : m
           sum = sum + (x1(i,1) - x2(i,1))^2; 
    end
    sim = exp(-sum/(2*(sigma^2)));
end
