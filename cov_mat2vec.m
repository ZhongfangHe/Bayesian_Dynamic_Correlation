

function a = cov_mat2vec(A)
% Inputs:
%    A: a n-by-n covariance matrix
% Outputs:
%    a: a (1+n)*n/2-by-1 vector stacking by rows of A, i.e. a_11, a_21, a_22, a_31, a_32, a_33, ....

n = size(A,1);
a = zeros((1+n)*n/2,1);

count = 1;
for i = 1:n
    tmp = A(i,1:i);
    a(count:count+i-1) = tmp';
    count = count + i;
end  
