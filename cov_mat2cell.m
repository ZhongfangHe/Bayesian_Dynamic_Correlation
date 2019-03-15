% Given a cell array {X1,X2,...,XT} with Xt being a covariance matrix, vectorize each matrix and stack the vectors into a matrix.

function xc = cov_mat2cell(xm)
% Inputs:
%   xm: a T-by-((1+n)*n/2) matrix stacking the vectorized covariance matrices.
% Outputs:
%   xc: a T-by-1 cell with each cell being a n-by-n covariance matrix.

[T,k] = size(xm);
n = sqrt(0.25+2*k) - 0.5;

xc = cell(T,1);
for t = 1:T
    xmt = xm(t,:)';
    xct = cov_vec2mat(xmt,n);
    xc{t} = xct;
end