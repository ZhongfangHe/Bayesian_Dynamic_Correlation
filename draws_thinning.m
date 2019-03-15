function y = draws_thinning(x,k)
% x = [x1 x2 x3 ... xn]
% y = [x1 x(1+k) x(1+2k) x(1+3k) ...]

% k should be fully divided by n=size(x,1), e.g. k = 5, 10, etc.

[n,p] = size(x);
m = n/k;
y = zeros(m,p);
for j = 1:p
    tmp = reshape(x(:,j),[k, m]);
    y(:,j) = tmp(1,:)';
end
    
