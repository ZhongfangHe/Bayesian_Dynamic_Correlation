% Perform the following posterior analyses:
% (1). Geweke test to check the convergence of draws;
% (2). Write and display the posterior means and 95% credible sets;
% (3). Write and display the acceptance rate of MH steps.

function [write_file_geweke, write_file_est, write_file_MH] = Posterior_Analysis(para_name, post_draws, MH_para_name, count_mat, write_path)
% Inputs:
%   para_name: a m-by-1 cell of the parameter names;
%   post_draws: a n-by-m matrix of the draws of the parameters;
%   MH_para_name: a k-by-1 cell of the names of the parameters that need MH steps;
%   count_mat: a n-by-k  matrix of the counts of MH acceptance;
%   write_path: a string of the path to write the files.
% Outputs:
%   write_file_geweke: a string of the file name that stores the Geweke test results;
%   write_file_est: a string of the file name that stores the posterior means and 95% credible sets;
%   write_file_MH: a string of the file name that stores the acceptance rates of MH steps.


%% Geweke convergence test
[gt_value, gt_cdf] = Geweke_Convergence_Test(post_draws);

lb = 0.025;
ub = 0.975;
fail_idx = and(gt_cdf <= lb, gt_cdf >= ub);

write_file_geweke = [write_path, 'Geweke_Test.csv'];
nof_para = length(para_name);
write_mat = cell(nof_para+1, 4);
write_mat(2:(nof_para+1),1) = para_name;
write_mat(1, 2:4) = {'Statistic', 'CDF', ['Failed(', num2str((ub-lb)*100), '%_CI)']};
write_mat(2:(nof_para+1),2:4) = num2cell([gt_value'  gt_cdf'   fail_idx']);
write_mat{1,1} = 'ParameterName';
text_file_write(write_file_geweke, write_mat, 1);



%% Posterior means and credible sets
post_draws_mean = mean(post_draws)';
lb_tail = 2.5;
ub_tail = 97.5;
post_draws_ci = prctile(post_draws, [lb_tail  ub_tail])';

write_file_est = [write_path, 'Posterior_Estimate.csv'];
nof_para = length(para_name);
write_mat = cell(nof_para+1, 4);
write_mat(2:(nof_para+1),1) = para_name;
write_mat(1, 2:4) = {'Mean', [num2str(lb_tail),'%_Percentile'], [num2str(ub_tail),'%_Percentile']};
write_mat(2:(nof_para+1),2:4) = num2cell([post_draws_mean   post_draws_ci]);
write_mat{1,1} = 'ParameterName';
text_file_write(write_file_est, write_mat, 1);

disp('Posterior means and credible sets:');
for j = 1:nof_para
    disp([para_name{j}, ' = ', num2str(post_draws_mean(j)), ' (', num2str(post_draws_ci(j,1)),...
        ', ', num2str(post_draws_ci(j,2)), '), Geweke test failure = ', num2str(fail_idx(j))]);
end
disp(' ');



%% Acceptance rates for MH steps
acceptance_rate = mean(count_mat)';

write_file_MH = [write_path, 'MH_Acceptance_Rate.csv'];
nof_MHpara = length(MH_para_name);
write_mat = cell(nof_MHpara+1, 2);
write_mat(2:(nof_MHpara+1),1) = MH_para_name;
write_mat(1, 2) = {'AcceptanceRate'};
write_mat(2:(nof_MHpara+1),2) = num2cell(acceptance_rate);
write_mat{1,1} = 'ParameterName';
text_file_write(write_file_MH, write_mat, 1);

disp('Acceptance rate for MH steps:');
for j = 1:nof_MHpara
    disp([MH_para_name{j}, ' = ', num2str(acceptance_rate(j))]);
end
disp(' ');




