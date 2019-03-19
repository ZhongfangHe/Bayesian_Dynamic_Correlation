% Estimate GDC model with normal innovation
% y_{i,t} = z_{i,t}' * w_i + sqrt(h_{i,t}) * u_{i,t}, i=1,2,...,m; t=1,2,...,T
%
% u_{i,t}~N(0,1), cov_{t-1}(u_{i,t},u_{j,t}) = r_{ij,t},
%
% GARCH: h_{i,t} = (1 - a_i - b_i) * h_i + a_i * h_{i,t-1} * (u_{i,t-1}^2) + b_i * h_{i,t-1},
% 
% GDC top: r_{ij,t} = (1 - e - f) * r_ij + e * q_{ij,t} + f * r_{ij,t-1}, where e+f=1
%
% GDC intermediate: q_{ij,t} = x_{ij,t}/sqrt(x_{ii,t} * x_{jj,t}) 
%
% GDC bottom: x_{ij,t} = (1 - c - d) * x_ij + c * u_{i,t-1} * u_{j,t-1} + d * x_{ij,t-1}, where c+d=1


clear;
rng(12345);


%% Inputs
% Specify directories of auxiliary functions and to write posterior analysis results
path_name = 'C:\Users\534474366\Documents\Research\DCC GARCH\Mar2019\auxiliary_functions'; %path of auxiliary functions 
write_path = 'C:\Users\534474366\Documents\Research\DCC GARCH\Mar2019\simulated_data\results_GDC\'; %path to write results; should already exist


% Specify info to read the target data (features are AR lags of targets)
read_path = 'C:\Users\534474366\Documents\Research\DCC GARCH\Mar2019\simulated_data\'; 
read_info.read_file = [read_path, 'Simulated_Data_GDC.xlsx']; 
read_info.read_sheet = 'Y';
read_info.read_cell = 'A2:B301'; %data and the snap date (not the first row)


% Specify the AR lags for each target
m = 2; %number of targets
vec_ARlags = cell(m,1); % features are AR lags (e.g. [1 3 12]')
vec_ARlags{1} = []; 
vec_ARlags{2} = []; 


% Specify the prior for the linear coef on the features
z_coef.prior_mean = cell(m,1); %means and covariance matrices of Gaussian priors for the linear coef of features
z_coef.prior_var = cell(m,1);
for j = 1:m
    nof_z = 1 + length(vec_ARlags{j});
    z_coef.prior_mean{j} = zeros(nof_z,1); %mean of Gaussian prior
    z_coef.prior_var{j} = 100 * eye(nof_z); %covariance matrix of Gaussian prior
end


% Specify the prior for GARCH parameters
garch.absum_p0 = 9 * ones(m,1); %prior: absum_i ~ Beta(absum_p0, absum_q0)
garch.absum_q0 = 1 * ones(m,1);
garch.ap_p0 = 1 * ones(m,1); %prior: ap_i ~ Beta(ap_p0, ap_q0)
garch.ap_q0 = 4 * ones(m,1);


% Specify the prior for GDC bottom parameters
gdc.cdsum_p0 = 9; %prior: cdsum ~ Beta(cdsum_p0, cdsum_q0)
gdc.cdsum_q0 = 1;
gdc.cp_p0 = 1; %prior: cp ~ Beta(cp_p0, cp_q0)
gdc.cp_q0 = 4;


% Specify the prior for GDC top parameters
gdc.efsum_p0 = 9; %prior: efsum ~ Beta(efsum_p0, efsum_q0)
gdc.efsum_q0 = 1;
gdc.ep_p0 = 9; %prior: ep ~ Beta(ep_p0, ep_q0)
gdc.ep_q0 = 1;


% Specify the number of MCMC draws
nof_burnin = 500; %number of burn-ins
nof_draws = 2000; %number of effective draws for posterior analysis


% Specify the parameters for MH steps
df.absum = [0.3  1]; %increase => lower acceptance rate
df.ap = [0.05  0.1]; %increase => lower acceptance rate
df.cdsum = 1; %increase => lower acceptance rate
df.cp = 1; %increase => lower acceptance rate
df.efsum = 1; %increase => lower acceptance rate
df.ep = 1; %increase => lower acceptance rate


% Specify the number of out-of-sample forecasts; 
nof_pred = 0; %nof_pred = 0 if only full-sample estimation is needed




%% Read raw data and process raw data to assemble targets and features (constant and AR lags)
% Targets could have different AR lags, but data of all targets start at 1+max(AR lag) and have the same length
addpath(genpath(path_name)); %add the path of the auxiliary functions
[y, z, nof_z_vec, snap_dt] = read_process_data(read_info, vec_ARlags, m);



%% Estimate the model with full-sample data or do out-of-sample forecasts by MCMC
if nof_pred == 0 %full-sample estimation
    [draws,count] = MCMC_GARCH_IGDC_normal(y, z, z_coef, garch, gdc, df, nof_burnin, nof_draws); %MCMC estimation; show progress every 1000 draws.
    save([write_path, 'results_IGDC.mat'], 'draws', 'count'); %use "results = load('results_DCC.mat') to load data.
    write_file = posterior_analysis_GDC_normal(draws, count, write_path, snap_dt, nof_z_vec); %write the results
else %out-of-sample forecast
    logp_vec = zeros(nof_pred,1); %store log predictive densities
    pred_para_est = zeros(nof_pred, 4+2*m);%store posterior means of parameters in forecasts
    pred_mh = zeros(nof_pred, 4+2*m); %store MH acceptance rates in forecasts
    nobs = size(y,1);
    for pred_j = 1:nof_pred
        Yt = y(1:(nobs-nof_pred+pred_j),:); %target data (the last row is for predictive density; others are for estimation)
        Yt_est = y(1:(nobs-nof_pred+pred_j-1),:);  
        
        Zt = cell(m,1);
        Zt_est = cell(m,1);
        for j = 1:m
            Zt{j} = z{j}(1:(nobs-nof_pred+pred_j),:);
            Zt_est{j} = z{j}(1:(nobs-nof_pred+pred_j-1),:);
        end %feature data (the last row is for predictive density; others are for estimation)
        
        [draws,count] = MCMC_GARCH_IGDC_normal(Yt_est, Zt_est, z_coef, garch, gdc, df, nof_burnin, nof_draws); %MCMC estimation; show progress every 1000 draws.

        logp_vec(pred_j) = loglike_pred_GDC_normal(Yt, Zt, draws); %compute log predictive density 

        pred_para_est(pred_j,:) = mean([draws.absum   draws.ap    draws.cdsum   draws.cp  draws.efsum   draws.ep]); %posterior mean of parameters at iteration j
        pred_mh(pred_j,:) = mean([count.absum  count.ap  count.cdsum  count.cp  count.efsum  count.ep]); %acceptance rate of MH steps at iteration j
        
        write_file_pred = info_pred_GDC_normal(logp_vec, pred_para_est, pred_mh, snap_dt, write_path); %write the results 
    end %end of prediction loop   
end





