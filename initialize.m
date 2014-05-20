function [theta info] = initialize(hiddenSize1, hiddenSize2, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize1 + visibleSize + hiddenSize2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize1, visibleSize) * 2 * r - r;
W2 = rand(hiddenSize2, hiddenSize1) * 2 * r - r;
W3 = rand(hiddenSize1, hiddenSize2) * 2 * r - r;
W4 = rand(visibleSize, hiddenSize1) * 2 * r - r; 

b1 = zeros(hiddenSize1, 1);
b2 = zeros(hiddenSize2, 1);
b3 = zeros(hiddenSize1, 1);
b4 = zeros(visibleSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
[theta info] = param2stack(W1,W2,W3,W4,b1,b2,b3,b4);
end

