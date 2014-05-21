function [cost,grad] = sparseAutoencoderCost(theta, info, visibleSize, hiddenSize1,hiddenSize2, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

[W1 W2 W3 W4 b1 b2 b3 b4] = stack2param(theta, info);
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
W3grad = zeros(size(W3));
W4grad = zeros(size(W4));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
b3grad = zeros(size(b3));
b4grad = zeros(size(b4));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
[ndims, m] = size(data);


z2 = zeros(hiddenSize1, m);
z3 = zeros(hiddenSize2, m);
z4 = zeros(hiddenSize1, m);
z5 = zeros(visibleSize, m);
a1 = zeros(ndims, m);
a2 = zeros(size(z2));
a3 = zeros(size(z3));
a4 = zeros(size(z4));
a5 = zeros(size(z5));
%autoencode use inputs as target values
y  = zeros(ndims, m);

a1 = data;
y = data;

deltaW1 = zeros(size(W1));
deltab1 = zeros(size(b1));
W1grad = zeros(size(W1));
b1grad = zeros(size(b1));

deltaW2 = zeros(size(W2));
deltab2 = zeros(size(b2));
W2grad = zeros(size(W2));
b2grad = zeros(size(b2));

%--------------------------

deltaW3 = zeros(size(W3));
deltab3 = zeros(size(b3));
W3grad = zeros(size(W3));
b3grad = zeros(size(b3));

deltaW4 = zeros(size(W4));
deltab4 = zeros(size(b4));
W4grad = zeros(size(W4));
b4grad = zeros(size(b4));


z2 = W1 * data + repmat(b1,1,m);
a2 = activation(z2);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = activation(z3);
z4 = W3 * a3 + repmat(b3,1,m);
a4 = activation(z4);
z5 = W4 * a4 + repmat(b4,1,m);
a5 = activation(z5);



% %compute the sparse rho
rho = (1. / m) * sum(a3, 2);
sp = sparsityParam;
sparsity_delta = -sp ./ rho + (1-sp) ./ (1-rho);
delta5 = -(y - a5) .* activationGrad(z5);
delta4 = (W4' * delta5) .* activationGrad(z4);
delta3 = (W3' * delta4 + beta*repmat(sparsity_delta, 1, m)) ...
    .* activationGrad(z3);
delta2 = (W2' * delta3) .* activationGrad(z2);


% % the sparse update term
% delta2 = ( W2' * delta3 + beta * repmat(sparsity_delta,1,m)) .* activationGrad(z2);

deltaW1 = delta2 * a1';
deltab1 = sum(delta2, 2);
deltaW2 = delta3 * a2';
deltab2 = sum(delta3, 2);
deltaW3 = delta4 * a3';
deltab3 = sum(delta4,2);
deltaW4 = delta5 * a4';
deltab4 = sum(delta5,2);


 

W1grad = (1. / m) * deltaW1 + lambda * W1;
b1grad = (1. / m) * deltab1;
W2grad = (1. / m) * deltaW2 + lambda * W2;
b2grad = (1. / m) * deltab2;
W3grad = (1. / m) * deltaW3 + lambda * W3;
b3grad = (1. / m) * deltab3;
W4grad = (1. / m) * deltaW4 + lambda * W4;
b4grad = (1. / m) * deltab4;

cost = (1. / m) * sum(0.5 * sum((a5 - y).^2)) + ...
    (lambda / 2.) * (sum(sum(W1.^2)) + sum(sum(W2.^2)) + ...
    sum(sum(W3.^2)) + sum(sum(W4.^2))) + ...
    beta * sum( sp*log(sp./rho) + (1-sp)*log((1-sp)./(1-rho)));


% % the cost with sparse term
% cost = (1. / m) * sum((1. / 2) * sum((a3 - y).^2)) + ...
%     (lambda / 2.) * (sum(sum(W1.^2)) + sum(sum(W2.^2))) + ...
%     beta * sum( sp*log(sp./rho) + (1-sp)*log((1-sp)./(1-rho)) );


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; W3grad(:) ; W4grad(:) ; b1grad(:); ...
    b2grad(:) ; b3grad(:) ; b4grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the activation function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function out = activation(x)

    % set the mu;
    
    mu = 0.5;
    
    if x < -mu
        out = -x - 0.5*mu;
    elseif x > mu
        out = x - 0.5*mu;
    else
        out = x.^2 / (2.0*mu);
    end
     
    
end


function grad = activationGrad(x)

     %set the mu;
     mu = 0.5;
     if  x < -mu
         grad = -1;
     elseif x > mu
         grad = 1;
     else
         grad = x / (mu*1.0);   
     end
    
end


