function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


S = 0;
for i = 1:m
  hyp = sigmoid(X(i, :) * theta);
  S = S - y(i)*log(hyp) - (1-y(i))*log(1-hyp);
  grad = grad + (hyp - y(i))* (X(i, :))';
endfor
Sreg = 0;
for j = 2:n
  Sreg = Sreg + theta(j)^2;
  grad(j) = grad(j) + lambda*theta(j);
endfor
J = S/m + lambda*Sreg/(2*m);
grad = grad/m;



% =============================================================

end
