% Compute log likelihood of VC model given the standardized innovation and VC parameters
% Assume standardized innovation ~ N(0,1)
% The long-run mean is approximated by normalized E(u_t*u_t')
% R_0 = normalized E(u_t*u_t')
% Q_1,...,Q_M = E(u_t*u_t'), Q_M starts to compute movinag average.

% To avoid unnecessary computation, input the normalized moving average directly.

function loglike = loglike_VC_normal(u, u2, a, b, x_mat)
% Inputs:
%   u: a n-by-m matrix of standardized innovations.
%   u2: a n-by-((m+1)*m/2) matrix of stacking vectorized u_t*u_t'.
%   a: a scalar of the MA coefficient.
%   b: a scalar of the AR coefficient.
%   x_mat: a n-by-((m-1)*m/2) matrix of stacking vectorized x_t (x_t = normalize(moving average of u_t*u_t'))
% Outputs:
%   loglike: a scalar of the loglikelihood (constant is removed)

[nobs,m] = size(u);


% %% compute Q_t
% MA_mat = MA_mat_construction(nobs,M);
% q_mat = VC_filter(u2, M, MA_mat);
% 
% 
% %% normalize Q_t to compute X_t
% x_mat = normalize_stacked_Q(q_mat);


%% compute R_t from X_t
mean_u2 = mean(u2);
R0_raw = cov_vec2mat(mean_u2',m);
R0 = matrix_normalize(R0_raw);
r_mat = VC_assemble(x_mat, a, b, R0);


%% loglikelihood
loglike = 0;
for t = 1:nobs
    ut = u(t,:)';
    rt = corr_vec2mat(r_mat(t,:)',m);
    det_rt = det(rt);
    rt_inv = rt\eye(m);
    loglike = loglike + log(det_rt) + ut' * rt_inv * ut;     
end
loglike = -0.5 * loglike;

