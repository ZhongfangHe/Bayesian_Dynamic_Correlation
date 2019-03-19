% Compute log likelihood of DCC model given the standardized innovation and DCC parameters
% Assume standardized innovation ~ N(0,1)
% The long-run mean is approximated by E(u_t*u_t')
% Q_1 = E(u_t*u_t')

function loglike = loglike_GDC_normal(u, u2, aq, bq, ar, br)
% Inputs:
%   u: a n-by-m matrix of standardized innovations.
%   u2: a n-by-((m+1)*m/2) matrix of stacking vectorized u_t*u_t'.
%   aq: a scalar of the MA coefficient for the bottom part.
%   bq: a scalar of the AR coefficient for the bottom part.
%   ar: a scalar of the MA coefficient for the top part.
%   br: a scalar of the AR coefficient for the top part.
% Outputs:
%   loglike: a scalar of the loglikelihood (constant is removed)

[nobs,m] = size(u);

%% compute Q_t
q_mat = DCC_filter(u2, aq, bq);


%% normalize Q_t to compute X_t
x_mat = normalize_stacked_Q(q_mat);


%% compute R_t from X_t
mean_u2 = mean(u2);
R0_raw = cov_vec2mat(mean_u2',m);
% mean_u2 = mean(x_mat);
% R0_raw = corr_vec2mat(mean_u2',m);
R0 = matrix_normalize(R0_raw);
r_mat = VC_assemble(x_mat, ar, br, R0);


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

