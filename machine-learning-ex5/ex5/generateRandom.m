function [rX , ry] = generateRandom(m,i,X,y)
    %vector of random indexes
   rX = zeros(size(X));
   ry = zeros(size(y));
   index = randsample(m,i);
    for c = 1 : i
      rX = X(index(c),:);
      ry = y(index(c),:);
    end
end