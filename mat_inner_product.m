% Given a matrix stacking u_t, compute the matrix stacking the vectorized u_t * u_t'

function u2 = mat_inner_product(u)
% Inputs:
%   u: a T-by-n matrix of data
% Outputs:
%   u2: a T-by-((n+1)*n/2) matrix with each row being the vectorized u_t * u_t'

[T,n] = size(u);
k = (n+1)*n/2;
u2 = zeros(T,k);
for t = 1:T
    u_t = u(t,:)';
    u2_t = u_t * u_t';
    u2(t,:) = cov_mat2vec(u2_t)';
end