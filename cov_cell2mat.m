% Given a cell array {X1,X2,...,XT} with Xt being a covariance matrix, vectorize each matrix and stack the vectors into a matrix.

function xm = cov_cell2mat(xc)
% Inputs:
%   xc: a T-by-1 cell with each cell being a n-by-n covariance matrix.
% Outputs:
%   xm: a T-by-((1+n)*n/2) matrix stacking the vectorized covariance matrices.

T = length(xc);
n = size(xc{1},1);
xm = zeros(T,(1+n)*n/2);
for t = 1:T
    xct = xc{t};
    xmt = cov_mat2vec(xct);
    xm(t,:) = xmt';
end