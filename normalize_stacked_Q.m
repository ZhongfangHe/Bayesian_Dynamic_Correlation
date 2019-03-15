% Given a matrix stacking vectorized Q_t, compute the corresponding matrix of normalized Q_t.

function x_mat = normalize_stacked_Q(q_mat)
% Inputs:
%   q_mat: a T-by-((n+1)*n/2) matrix of stacking vectorized Q_t
% Outputs:
%   x_mat: a T-by-((n-1)*n/2) matrix of normalized Q_t (columns of ones are removed) 

[T,K] = size(q_mat);
M = sqrt(2 * K + 0.25) - 0.5; %size of q_t
x_mat = zeros(T,(M-1)*M/2);
for t = 1:T
    q_t_vec = q_mat(t,:)';
    q_t = cov_vec2mat(q_t_vec, M);

    x_t = matrix_normalize(q_t);
    
    x_t_vec = corr_mat2vec(x_t);
    x_mat(t,:) = x_t_vec';
end
