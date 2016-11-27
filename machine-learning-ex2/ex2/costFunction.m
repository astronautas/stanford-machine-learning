function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
sigmoid_values = sigmoid(theta' * X');

first_log = log(sigmoid_values);
second_log = log(1 - sigmoid_values);

first_member = -y' * log(sigmoid_values)';
second_member = (1 - y)' * log(1 - sigmoid_values)';

member_subs = first_member - second_member;
J = sum(member_subs(:)) / m;

partials = theta;

% GRADIENT
for j = 1:length(theta)
    partial = sum( ((sigmoid_values' - y)'*X(:, j)) ) / m;
    partials(j) = partial;
end

grad = partials;
% =============================================================

end
