% test convergence by Geweke (1992).
% use the Newey-West method to estimate the long-run variance with the truncation lag floor(4 * (T/100)^(2/9)).
% "gt_cdf" should be within a reasonable range, e.g. (0.05  0.95)

function [gt_value, gt_cdf] = Geweke_Convergence_Test(x)
% Inputs:
%   x: a n-by-m matrix of posterior draws for m parameters
% Outputs:
%   gt_value: a m-by-1 vector of Geweke test statistics
%   gt_cdf: a m-by-1 vector of the CDF of Geweke test statistics

[n,m] = size(x);

n1 = round(0.1*n);
n2 = round(0.4*n);

x1 = x(1:n1,:);
x2 = x(n-n2+1:n,:);

mu_x1 = mean(x1);
mu_x2 = mean(x2);

u1 = x1 - ones(n1,1) * mu_x1;
u2 = x2 - ones(n2,1) * mu_x2;

k1 = floor(4 * ((n1/100)^(2/9)));
k2 = floor(4 * ((n2/100)^(2/9)));

var1 = var(x1);
var2 = var(x2);
for i = 1:m
    tmp = var1(i);
    for j = 1:k1
        tmp = tmp + 2 * (1 - j / (k1 + 1)) * (u1(1:n1-j,i)' * u1(1+j:n1,i)) / (n1-j);
    end
    var1(i) = tmp;
    
    tmp = var2(i);
    for j = 1:k2
        tmp = tmp + 2 * (1 - j / (k2 + 1)) * (u2(1:n2-j,i)' * u2(1+j:n2,i)) / (n2-j);
    end
    var2(i) = tmp;       
end

gt_value = (mu_x1 - mu_x2) ./ sqrt((var1 / n1 + var2 / n2));
gt_cdf = normcdf(gt_value);
    
    




