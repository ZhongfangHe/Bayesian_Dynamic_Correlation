% Given a cell array {X1,X2,...,XT} with Xt being a correlation matrix, vectorize each matrix and stack the vectors into a matrix.

function xm = corr_cell2mat(xc)
% Inputs:
%   xc: a T-by-1 cell with each cell being a n-by-n correlation matrix.
% Outputs:
%   xm: a T-by-((n-1)*n/2) matrix stacking the vectorized correlation matrices.

T = length(xc);
n = size(xc{1},1);
xm = zeros(T,(n-1)*n/2);
for t = 1:T
    xct = xc{t};
    xmt = corr_mat2vec(xct);
    xm(t,:) = xmt';
end