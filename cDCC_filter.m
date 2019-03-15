% Q_t from cDCC with targeting (set initial value Q_1 = E(u2))
% q_{ij,t} = (1 - a - b)*s_{ij} + a*sqrt(q_{ii,t-1}*q_{jj,t-1})*(u_{i,t-1}*u_{j,t-1}) + b*q_{ij,t-1}
% s_{ii} = 1


function q_mat = cDCC_filter(u, u2, a, b)
% Inputs:
%   u: a T-by-n matrix of residual u_t
%   u2: a T-by-((n+1)*n/2) matrix of vectorized u_t * u_t'
%   a: a scalar of coef on u_{t-1}*u_{t-1}'
%   b: a scalar of coef on Q_{t-1}
% Outputs:
%   q_mat: a T-by-((n+1)*n/2) matrix of vectorized Q_t

[T,n] = size(u);


%% Compute diagonal elements of Q_t and the scaled residual u_t .* sqrt(diagonal(Q_t))
q_diagonal_mat = ones(T,n);
const = 1 - a - b;
idx_diagonal = cumsum(1:n);
u2_diagonal = u2(:,idx_diagonal);
u_star = u;
for t = 2:T
    q_tm1 = q_diagonal_mat(t-1,:)';
    u2_diagonal_tm1 = u2_diagonal(t-1,:)';
    q_t = const + a * (q_tm1 .* u2_diagonal_tm1) + b * q_tm1;
    q_diagonal_mat(t,:) = q_t';
    
    u_star(t,:) = u(t,:) .* sqrt(q_t)';  
end


%% Compute the target S
%S = (u_star' * u_star) / T;


%% Compute u2_star
u2_star = mat_inner_product(u_star);


%% Iterate Q_t (diagonal elements of Q_t are recomputed to ensure pd)
mean_u2 = mean(u2_star);

const = (1 - a - b) * mean_u2;
sig2_1 = mean_u2;

[coef_mat_x, coef_vec_y1, coef_vec_s] = GARCH_b_coef(b, T);
q_mat = coef_vec_s * const + a * coef_mat_x * u2 + coef_vec_y1 * sig2_1;




