% Estimate DCC model with normal innovation
% y_{i,t} = z_{i,t}' * w_i + sqrt(h_{i,t}) * u_{i,t}, i=1,2,...,m; t=1,2,...,T
%
% u_{i,t}~N(0,1), cov_{t-1}(u_{i,t},u_{j,t}) = r_{ij,t},
%
% GARCH: h_{i,t} = (1 - a_i - b_i) * h_i + a_i * h_{i,t-1} * (u_{i,t-1}^2) + b_i * h_{i,t-1},
% 
% DCC top: r_{ij,t} = x_{ij,t}/sqrt(x_{ii,t} * x_{jj,t})
%
% DCC bottom: x_{ij,t} = (1 - c_ij - d_ij) * x_ij + c_ij * u_{i,t-1} * u_{j,t-1} + d_ij * x_{ij,t-1},


function [draws,count] = MCMC_GARCH_DCC_normal(y, z, z_coef, garch, dcc, df, nof_burnin, nof_draws)
% Inputs
%   y: a T-by-m matrix of target data by column
%   z: a m-by-1 cell of features data with cell i being a T-by-k_i matrix, i = 1,2,...,m
%   z_coef: a structure with 2 fields for the Gaussian prior of the linear coef on features:
%      z_coef.prior_mean: a m-by-1 cell of the mean of Gaussian prior; cell i is a k_i-by-1 vector
%      z_coef.prior_var: a m-by-1 cell of the covariance matrix of Gaussian prior; cell i is a k_i-by-k_i matrix 
%   garch: a structure with 4 fields for the Beta priors of GARCH parameters:
%      garch.absum_p0: a m-by-1 vector of the first parameter of Beta prior for a+b
%      garch.absum_q0: a m-by-1 vector of the second parameter of Beta prior for a+b
%      garch.ap_p0: a m-by-1 vector of the first parameter of Beta prior for a/(a+b)
%      garch.ap_q0: a m-by-1 vector of the second parameter of Beta prior for a/(a+b)      
%   dcc: a structure with 4 fields for the Beta priors of DCC parameters:
%      dcc.cdsum_p0: a scalar of the first parameter of Beta prior for c+d
%      dcc.cdsum_q0: a scalar of the second parameter of Beta prior for c+d
%      dcc.cp_p0: a scalar of the first parameter of Beta prior for c/(c+d)
%      dcc.cp_q0: a scalar of the second parameter of Beta prior for c/(c+d) 
%   df: a structure with 4 fields for the proposal of MH steps:
%      df.absum: a m-by-1 vector of the stdev of the MH proposal for a+b
%      df.ap: a m-by-1 vector of the stdev of the MH proposal for a/(a+b)
%      df.cdsum: a scalar of the stdev of the MH proposal for c+d
%      df.cp: a scalar of the stdev of the MH proposal for c/(c+d) 
%   nof_burnin: a scalar of the number of burn-ins
%   nof_draws: a scalar of the number of subsequent MCMC draws for analysis (total number of MCMC draws = nof_burnin + nof_draws)
% Outputs:
%   draws: a structure with 9 fields for the MCMC results:
%      draws.w: a m-by-1 cell with cell i being a n-by-k_i matrix of draws for linear coef of features
%      draws.absum: a n-by-m matrix of draws for a+b
%      draws.ap: a n-by-m matrix of draws for a/(a+b)
%      draws.a: a n-by-m matrix of draws for a
%      draws.b: a n-by-m matrix of draws for b
%      draws.cdsum: a n-by-1 vector of draws for c+d
%      draws.cp: a n-by-1 vector of draws for c/(c+d)
%      draws.c: a n-by-1 vector of draws for c
%      draws.d: a n-by-1 vector of draws for d
%   count: a structure with 4 fields for the acceptance counts of MH steps:
%      count.absum: a n-by-m matrix of acceptance counts for a+b
%      count.ap: a n-by-m matrix of acceptance counts for a/(a+b)
%      count.cdsum: a n-by-1 vector of acceptance counts for c+d
%      count.cp: a n-by-1 vector of acceptance counts for c/(c+d)


%% Get data sizes
[nobs,m] = size(y);
nof_z_vec = zeros(m,1);
for j = 1:m
    nof_z_vec(j) = size(z{j},2);
end


%% Set prior for GARCH coef
absum = zeros(m,1);
ap = zeros(m,1);
for j = 1:m
    absum(j) = betarnd(garch.absum_p0(j), garch.absum_q0(j));
    ap(j) = betarnd(garch.ap_p0(j), garch.ap_q0(j));
end
a = ap .* absum;
b = absum - a;
uh_dot2 = (y-kron(ones(nobs,1),mean(y))).^2; %matrix of uh.^2, where uh = sqrt(h)*u
h = zeros(nobs,m); %matrix of variances
h_half = zeros(nobs,m); %matrix of stdev
for j = 1:m
    h(:,j) = GARCH_filter(uh_dot2(:,j), a(j), b(j));
    h_half(:,j) = sqrt(h(:,j));
end


%% Set prior for linear coef w (multivariate normal)
z_coef.prior_mean_addon = cell(m,1);
z_coef.prior_var_inv = cell(m,1);
w = cell(m,1); %cell of linear coefficients
uh = zeros(nobs,m); %matrix of sqrt(h)*u 
u = zeros(nobs,m);
for j = 1:m
    nof_z = nof_z_vec(j);
    z_coef.prior_var_inv{j} = z_coef.prior_var{j}\eye(nof_z);
    z_coef.prior_mean_addon{j} = z_coef.prior_var_inv{j} * z_coef.prior_mean{j};    
    
    y_adj = y(:,j)./h_half(:,j);
    z_adj = z{j}./(h_half(:,j) * ones(1,nof_z)); 
    
    z_coef.post_var_inv = z_coef.prior_var_inv{j} + z_adj' * z_adj;
    z_coef.post_var = z_coef.post_var_inv\eye(nof_z);
    z_coef.post_mean = z_coef.post_var * (z_coef.prior_mean_addon{j} + z_adj' * y_adj);
    w{j} = mvnrnd(z_coef.post_mean, z_coef.post_var)';
    
    uh(:,j) = y(:,j) - z{j} * w{j};
    u(:,j) = uh(:,j)./h_half(:,j);
end
uh_dot2 = uh.^2; %uh_t.^2 for GARCH


%% Set prior for DCC coef
cdsum = betarnd(dcc.cdsum_p0, dcc.cdsum_q0);
cp = betarnd(dcc.cp_p0, dcc.cp_q0);


%% MCMC draws
nof_total = nof_burnin + nof_draws;
draws.w = cell(m,1);
for j = 1:m
    draws.w{j} = zeros(nof_draws,nof_z_vec(j));
end
draws.absum = zeros(nof_draws,m);
draws.ap = zeros(nof_draws,m);
draws.a = zeros(nof_draws,m);
draws.b = zeros(nof_draws,m);
draws.cdsum = zeros(nof_draws,1);
draws.cp = zeros(nof_draws,1);
draws.c = zeros(nof_draws,1);
draws.d = zeros(nof_draws,1);
draws.h_mean = zeros(nobs,m);
draws.h_std = zeros(nobs,m);
draws.corr_mean = zeros(nobs,(m-1)*m/2);
draws.corr_std = zeros(nobs,(m-1)*m/2);

count.absum = zeros(nof_draws,m);
count.ap = zeros(nof_draws,m);
count.cdsum = zeros(nof_draws,1);
count.cp = zeros(nof_draws,1);

tic;
for i = 1:nof_total
    % GARCH and linear coef over loop
    for j = 1:m
        % Prepare the data
        uh_j_dot2 = uh_dot2(:,j);

        
        % MH step for absum_j
        para = absum(j);
        mh = df.absum(j);
        mh_half = sqrt(mh);
        tmp = trandn(-para/mh_half, (1-para)/mh_half);
        para_prop = para + mh_half * tmp;

        log_prior = (garch.absum_p0(j)-1)*log(para) + (garch.absum_q0(j)-1)*log(1-para);
        log_prior_prop = (garch.absum_p0(j)-1)*log(para_prop) + (garch.absum_q0(j)-1)*log(1-para_prop);

        log_like = loglike_GARCH_normal(uh_j_dot2, para*ap(j), para-para*ap(j));
        log_like_prop = loglike_GARCH_normal(uh_j_dot2, para_prop*ap(j), para_prop-para_prop*ap(j));
        
        log_Q_prop = -log(normcdf((1-para)/mh_half) - normcdf(-para/mh_half));
        log_Q = -log(normcdf((1-para_prop)/mh_half) - normcdf(-para_prop/mh_half));

        log_accp_prob = (log_prior_prop + log_like_prop - log_Q_prop) - (log_prior + log_like - log_Q);
        accp_prob = exp(log_accp_prob);
        tmp = rand(1,1);
        if tmp < accp_prob
            absum(j) = para_prop;
            if i > nof_burnin
                count.absum(i-nof_burnin,j) = 1;
            end
        end        
        if i > nof_burnin
            draws.absum(i-nof_burnin,j) = absum(j);
        end


        % MH step for ap_j
        para = ap(j);
        mh = df.ap(j);
        mh_half = sqrt(mh);
        tmp = trandn(-para/mh_half, (1-para)/mh_half);
        para_prop = para + mh_half * tmp;

        log_prior = (garch.ap_p0(j)-1)*log(para) + (garch.ap_q0(j)-1)*log(1-para);
        log_prior_prop = (garch.ap_p0(j)-1)*log(para_prop) + (garch.ap_q0(j)-1)*log(1-para_prop);

        log_like = loglike_GARCH_normal(uh_j_dot2, para*absum(j), absum(j)-para*absum(j));
        log_like_prop = loglike_GARCH_normal(uh_j_dot2, para_prop*absum(j), absum(j)-para_prop*absum(j));

        log_Q_prop = -log(normcdf((1-para)/mh_half) - normcdf(-para/mh_half));
        log_Q = -log(normcdf((1-para_prop)/mh_half) - normcdf(-para_prop/mh_half));

        log_accp_prob = (log_prior_prop + log_like_prop - log_Q_prop) - (log_prior + log_like - log_Q);
        accp_prob = exp(log_accp_prob);
        tmp = rand(1,1);
        if tmp < accp_prob
            ap(j) = para_prop;
            if i > nof_burnin
                count.ap(i-nof_burnin,j) = 1;
            end
        end        
        if i > nof_burnin
            draws.ap(i-nof_burnin,j) = ap(j);
        end


        % Calculate GARCH variances
        a(j) = absum(j) * ap(j);
        b(j) = absum(j) - a(j);       
        h(:,j) = GARCH_filter(uh_j_dot2, a(j), b(j));
        h_half(:,j) = sqrt(h(:,j));
        if i > nof_burnin
            draws.a(i-nof_burnin,j) = a(j);
            draws.b(i-nof_burnin,j) = b(j);
            draws.h_mean(:,j) = draws.h_mean(:,j) + h(:,j);
            draws.h_std(:,j) = draws.h_std(:,j) + h(:,j).^2;            
        end         


        % Gibbs step for w_j
        nof_z = nof_z_vec(j);
        y_adj = y(:,j)./h_half(:,j);
        z_adj = z{j}./(h_half(:,j) * ones(1,nof_z)); 
        z_coef.post_var_inv = z_coef.prior_var_inv{j} + z_adj' * z_adj;
        z_coef.post_var = z_coef.post_var_inv\eye(nof_z);
        z_coef.post_mean = z_coef.post_var * (z_coef.prior_mean_addon{j} + z_adj' * y_adj);
        w{j} = mvnrnd(z_coef.post_mean, z_coef.post_var)';
        if i > nof_burnin
            draws.w{j}(i-nof_burnin,:) = w{j}';
        end        
        
        
        % Update the residuals
        uh(:,j) = y(:,j) - z{j} * w{j}; %sqrt(h)*u; non-standardized residual
        u(:,j) = uh(:,j)./h_half(:,j); %standardized residual
        uh_dot2(:,j) = uh(:,j).^2; %uh.^2 for GARCH
        
    end %end of GARCH loop
    u2 = mat_inner_product(u); %vectorize u_t*u_t'
       
    
    % DCC parameters
    % MH step for cdsum
    para = cdsum;
    mh = df.cdsum;
    mh_half = sqrt(mh);
    tmp = trandn(-para/mh_half, (1-para)/mh_half);
    para_prop = para + mh_half * tmp;

    log_prior = (dcc.cdsum_p0-1)*log(para) + (dcc.cdsum_q0-1)*log(1-para);
    log_prior_prop = (dcc.cdsum_p0-1)*log(para_prop) + (dcc.cdsum_q0-1)*log(1-para_prop);

    log_like = loglike_DCC_normal(u, u2, para*cp, para-para*cp);
    log_like_prop = loglike_DCC_normal(u, u2, para_prop*cp, para_prop-para_prop*cp);

    log_Q_prop = -log(normcdf((1-para)/mh_half) - normcdf(-para/mh_half));
    log_Q = -log(normcdf((1-para_prop)/mh_half) - normcdf(-para_prop/mh_half));

    log_accp_prob = (log_prior_prop + log_like_prop - log_Q_prop) - (log_prior + log_like - log_Q);
    accp_prob = exp(log_accp_prob);
    tmp = rand(1,1);
    if tmp < accp_prob
        cdsum = para_prop;
        if i > nof_burnin
            count.cdsum(i-nof_burnin) = 1;
        end
    end        
    if i > nof_burnin
        draws.cdsum(i-nof_burnin) = cdsum;
    end  
    
    
     % MH step for cp
    para = cp;
    mh = df.cp;
    mh_half = sqrt(mh);
    tmp = trandn(-para/mh_half, (1-para)/mh_half);
    para_prop = para + mh_half * tmp;

    log_prior = (dcc.cp_p0-1)*log(para) + (dcc.cp_q0-1)*log(1-para);
    log_prior_prop = (dcc.cp_p0-1)*log(para_prop) + (dcc.cp_q0-1)*log(1-para_prop);

    log_like = loglike_DCC_normal(u, u2, para*cdsum, cdsum-para*cdsum);
    log_like_prop = loglike_DCC_normal(u, u2, para_prop*cdsum, cdsum-para_prop*cdsum);

    log_Q_prop = -log(normcdf((1-para)/mh_half) - normcdf(-para/mh_half));
    log_Q = -log(normcdf((1-para_prop)/mh_half) - normcdf(-para_prop/mh_half));

    log_accp_prob = (log_prior_prop + log_like_prop - log_Q_prop) - (log_prior + log_like - log_Q);
    accp_prob = exp(log_accp_prob);
    tmp = rand(1,1);
    if tmp < accp_prob
        cp = para_prop;
        if i > nof_burnin
            count.cp(i-nof_burnin) = 1;
        end
    end        
    if i > nof_burnin
        draws.cp(i-nof_burnin) = cp;
    end    
    
    
    % Calculate DCC correlations
    c = cdsum * cp;
    d = cdsum - c;   
    q_mat = DCC_filter(u2, c, d);
    r_mat = normalize_stacked_Q(q_mat);
    if i > nof_burnin
        draws.c(i-nof_burnin) = c;
        draws.d(i-nof_burnin) = d;
        draws.corr_mean = draws.corr_mean + r_mat;
        draws.corr_std = draws.corr_std + r_mat.^2;         
    end      
   

    % Display message to check progress
    if round(i/1000) == (i/1000)
        disp([num2str(i), ' draws have completed!']);
        toc;
    end
end


%% Compute estimated variances and correlations
draws.corr_mean = draws.corr_mean/nof_draws;
draws.corr_std = sqrt(draws.corr_std/nof_draws - draws.corr_mean.^2);
draws.corr_lb = draws.corr_mean - 2 * draws.corr_std;
draws.corr_ub = draws.corr_mean + 2 * draws.corr_std;

draws.h_mean = draws.h_mean/nof_draws;
draws.h_std = sqrt(draws.h_std/nof_draws - draws.h_mean.^2);
draws.h_lb = draws.h_mean - 2 * draws.h_std;
draws.h_ub = draws.h_mean + 2 * draws.h_std;



    
    
