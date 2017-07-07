function test_error = testCurve (X,y,X1test, ytest,lambda)
theta = trainLinearReg(X,y,lambda);
    [test_error, ~] = linearRegCostFunction(X1test,ytest,theta,0);
end