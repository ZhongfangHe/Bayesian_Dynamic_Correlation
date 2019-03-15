% Compute log likelihood of DCC model given the standardized innovation and DCC parameters
% Assume standardized innovation ~ N(0,1)
% The long-run mean is approximated by E(u_t*u_t')
% Q_1 = E(u_t*u_t')

function loglike = loglike_DCC_normal(u, u2, a, b)
% Inputs:
%   u: a n-by-m matrix of standardized innovations.
%   u2: a n-by-((m+1)*m/2) matrix of stacking vectorized u_t*u_t'.
%   a: a scalar of the MA coefficient.
%   b: a scalar of the AR coefficient.
% Outputs:
%   loglike: a scalar of the loglikelihood (constant is removed)

[nobs,m] = size(u);

%% compute Q_t
q_mat = DCC_filter(u2, a, b);

%% normalize Q_t to compute R_t
r_mat = normalize_stacked_Q(q_mat);

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

