% Compute the log predictive density for out-of-sample forecast
% Given target data {y_1, y_2, ..., y_T} and feature data {z_1, z_2, ..., z_T}, compute p(y_T|y_{1:T-1}, z_{1:T}) 

function logp = loglike_pred_VC_normal(Yt, Zt, draws, M)
% Inputs:
%   Yt: a T-by-m matrix of target data by column
%   Zt: a m-by-1 cell of feature data with cell j being a T-by-k_j matrix of feature data for target j
%   draws: a structure with 9 fields for the MCMC results:
%      draws.w: a m-by-1 cell with cell i being a n-by-k_i matrix of draws for linear coef of features
%      draws.absum: a n-by-m matrix of draws for a+b
%      draws.ap: a n-by-m matrix of draws for a/(a+b)
%      draws.a: a n-by-m matrix of draws for a
%      draws.b: a n-by-m matrix of draws for b
%      draws.efsum: a n-by-1 vector of draws for e+f
%      draws.ep: a n-by-1 vector of draws for e/(e+f)
%      draws.e: a n-by-1 vector of draws for e
%      draws.f: a n-by-1 vector of draws for f
%   M: a scalar of the MA length
% Outputs:
%   logp: a scalar of the log predictive density for y_T

[nobs, m] = size(Yt);
nof_draws = size(draws.a,1);

p_vec = zeros(nof_draws,1);
for i = 1:nof_draws
    % Compute the standardized and non-standardized innovations
    u = zeros(nobs,m);
    h = zeros(nobs,m);
    mu = zeros(nobs,m);
    for j = 1:m
        % Compute the mean
        w = draws.w{j}(i,:)';
        z = Zt{j};        
        mu(:,j) = z * w;        
        
        % Compute the non-standardized innovations
        y = Yt(:,j);
        uh = y - mu(:,j);
        
        % Compute GARCH variances
        uh_dot2 = uh.^2;
        a = draws.a(i,j);
        b = draws.b(i,j);
        h(:,j) = GARCH_filter(uh_dot2, a, b);
        
        % Compute the standardized innovations
        u(:,j) = uh ./ sqrt(h(:,j)); 
    end
    
    
    % VC correlations
    u2 = mat_inner_product(u); %vectorize u_t*u_t'
    
    MA_mat = MA_mat_construction(nobs,M);
    q_mat = VC_filter(u2, M, MA_mat);
    x_mat = normalize_stacked_Q(q_mat); % normalize the MA of u_t*u_t'
    
    mean_u2 = mean(u2);
    R0_raw = cov_vec2mat(mean_u2',m);
    R0 = matrix_normalize(R0_raw); %starting value for R_t    
    
    e = draws.e(i);
    f = draws.f(i);
    r_mat = VC_assemble(x_mat, e, f, R0); %stacking vectorized correlation matrices
    
    % Compute the covariance matrix of predictive density
    htp1 = h(nobs,:)';
    rtp1 = r_mat(nobs,:)';
    
    htp1_half_mat = diag(sqrt(htp1));
    rtp1_mat = corr_vec2mat(rtp1, m);
    cov_tp1 = htp1_half_mat * rtp1_mat * htp1_half_mat;
    
    % Compute the mean of predictive density
    mu_tp1 = mu(nobs,:)';
    
    % Compute log predictive density
    p_vec(i) = mvnpdf(Yt(nobs,:)', mu_tp1, cov_tp1); 
end

logp = log(mean(p_vec));
    





