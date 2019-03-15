% Generate the matrix for moving average [ones(1,M) zeros(1,T-M); zeros(1,1)  ones(1,M) zeros(1,T-M-1); ... ; zeros(1,T-M) ones(1,M)].
% y_t = (x_{t-1} + ... + x_{t-M})/M, t = 1,2,...,T
% use for VC

function MA_mat = MA_mat_construction(T,M)
% Inputs:
%   T: a scalar of the full-length
%   M: a scalar of the window length
% Outputs:
%   MA_mat: a (T-M)-by-T matrix to compute moving average

MA_mat = zeros(T-M,T);

tmp = ones(1,M);
for i = 1:(T-M)
    MA_mat(i,i:(i+M-1)) = tmp;
end