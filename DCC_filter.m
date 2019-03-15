% Q_t from DCC with targeting (set initial value Q_1 = E(u2))

function q_mat = DCC_filter(u2, a, b)
% Inputs:
%   u2: a T-by-((n+1)*n/2) matrix of vectorized u_t * u_t'
%   a: a scalar of coef on u_{t-1}*u_{t-1}'
%   b: a scalar of coef on Q_{t-1}
% Outputs:
%   q_mat: a T-by-((n+1)*n/2) matrix of vectorized Q_t

T = length(u2);
mean_u2 = mean(u2);

const = (1 - a - b) * mean_u2;
sig2_1 = mean_u2;

[coef_mat_x, coef_vec_y1, coef_vec_s] = GARCH_b_coef(b, T);
q_mat = coef_vec_s * const + a * coef_mat_x * u2 + coef_vec_y1 * sig2_1;