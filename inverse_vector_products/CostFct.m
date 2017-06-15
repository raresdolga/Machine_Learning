function J = CostFct(A, x,b)
J = norm(A*x-b)^2;
end
function A_inv_b = inverseVevtorProduct(A, b, x_init, alpha)
  cost = norm(A*x_init-b)^2 ;
  while(cost>10^(-6))
  x_init = x_init - alpha*2*A*(A*x_init - b);
  cost = norm(A*x_init-b)^2 ;
  end
  A_inv_b = x_init;
end