% Posterior analysis for cDCC estimates
% Write estimated variances/correlations, posterior means/95% CIs, acceptance rate of MH steps, Geweke tests into csv files.
% Display posterior means/95% CIs, acceptance rate of MH steps, Geweke tests 


function write_file = posterior_analysis_cDCC_normal(draws, count, write_path, snap_dt, nof_z_vec)
% Inputs:
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
%   write_path: a string of the path to write the files.
%   snap_dt: a T-by-1 cell of snap date or [] if no snap date
%   nof_z_vec: a m-by-1 vector of the number of features for each target
% Outputs:
%   write_file: a structure with 5 fields of write file names:
%      write_file.write_file_GARCH: a string of the file name to write variances
%      write_file.write_file_corr: a string of the file name to write correlations
%      write_file.write_file_Geweke: a string of the file name to write Geweke test results
%      write_file.write_file_est: a string of the file name to write posterior estimates
%      write_file.write_file_MH: a string of the file name to write MH acceptance rate



%% write estimated variances
[nobs,m] = size(draws.h_mean);
title_mean = cell(m,1);
title_ub = cell(m,1);
title_lb = cell(m,1);
for j = 1:m
    title_mean{j} = ['var',num2str(j),'_mean'];
    title_ub{j} = ['var',num2str(j),'_lb'];
    title_lb{j} = ['var',num2str(j),'_ub'];
end
write_file.write_file_GARCH = [write_path, 'GARCH_variance.csv'];
if isempty(snap_dt)
    ind_date = 0;
    write_mat = cell(nobs+1, 3*m);
    write_mat(1,:) = [title_mean; title_lb; title_ub]';
    write_mat(2:(nobs+1),:) = num2cell([draws.h_mean  draws.h_lb  draws.h_ub]);
else
    ind_date = 1;
    nobs_raw = length(snap_dt);
    write_mat = cell(nobs+1, 3*m+1);
    write_mat{1,1} = 'Date';
    write_mat(1,2:(3*m+1)) = [title_mean; title_lb; title_ub]';
    write_mat(2:(nobs+1),2:(3*m+1)) = num2cell([draws.h_mean  draws.h_lb  draws.h_ub]);
    write_mat(2:(nobs+1),1) = snap_dt((nobs_raw-nobs+1):nobs_raw);
end
text_file_write(write_file.write_file_GARCH, write_mat, ind_date);



%% write estimated correlations
name_R_mean = cell(1,(m-1)*m/2);
name_R_ub = cell(1,(m-1)*m/2);
name_R_lb = cell(1,(m-1)*m/2);
countr = 1;
for i = 2:m
    for j = 1:(i-1)
        name_R_mean{countr} = ['R_',num2str(i),'_',num2str(j),'_mean'];
        name_R_ub{countr} = ['R_',num2str(i),'_',num2str(j),'_ub'];
        name_R_lb{countr} = ['R_',num2str(i),'_',num2str(j),'_lb'];
        countr = countr + 1;
    end
end
write_file.write_file_corr = [write_path, 'correlation.csv'];
if isempty(snap_dt)
    write_mat = cell(nobs+1, 3*(m-1)*m/2);
    write_mat(1,:) = [name_R_mean, name_R_ub, name_R_lb];
    write_mat(2:(nobs+1),:) = num2cell([draws.corr_mean  draws.corr_lb  draws.corr_ub]);
else
    write_mat = cell(nobs+1, 3*(m-1)*m/2+1);
    write_mat{1,1} = 'Date';
    write_mat(1,2:(3*(m-1)*m/2+1)) = [name_R_mean, name_R_ub, name_R_lb];
    write_mat(2:(nobs+1),2:(3*(m-1)*m/2+1)) = num2cell([draws.corr_mean  draws.corr_lb  draws.corr_ub]);
    write_mat(2:(nobs+1),1) = snap_dt((nobs_raw-nobs+1):nobs_raw);
end
text_file_write(write_file.write_file_corr, write_mat, ind_date);



%% Compute Geweke test, posterior mean/ci, acceptance rate
% Assemble names of parameters and matrix of draws
name_a = cell(m,1);
name_b = cell(m,1);
name_w = cell(sum(nof_z_vec),1);
post_draws = [draws.a   draws.b    draws.c   draws.d];
for j = 1:m
    name_a{j} = ['a_', num2str(j)];
    name_b{j} = ['b_', num2str(j)];
    nof_z = nof_z_vec(j);
    name_w{j} = cell(nof_z,1);
    for jj = 1:nof_z
        name_w{j}{jj} = ['w_', num2str(j), '_', num2str(jj)];
    end
    post_draws = [post_draws  draws.w{j}]; %matrix of all draws
end
para_name = [name_a; name_b; {'c'}; {'d'}];
for j = 1:m
    para_name = [para_name; name_w{j}]; 
end %name of the parameters


% Assemble names of parameters that did MH steps and matrix of counts of acceptance
name_absum = cell(m,1);
name_ap = cell(m,1);
for j = 1:m
    name_absum{j} = ['absum_', num2str(j)];
    name_ap{j} = ['ap_', num2str(j)];
end
MH_para_name = [name_absum; name_ap; {'cdsum'}; {'cp'}]; %name of the parameters that did MH steps
count_mat = [count.absum   count.ap  count.cdsum  count.cp]; %matrix of the counts of MH acceptance


% Write and display the Geweke test, posterior mean/ci, acceptance rate of MH  
[write_file.write_file_geweke, write_file.write_file_est, write_file.write_file_MH] = Posterior_Analysis(para_name, post_draws, ...
    MH_para_name, count_mat, write_path);


