

function a = corr_mat2vec(A)
% Inputs:
%    A: a n-by-n correlation matrix
% Outputs:
%    a: a (n-1)*n/2-by-1 vector stacking by rows of A, i.e. a_21, a_31, a_32, a_41, a_42, a_43, ....

n = size(A,1);
a = zeros((n-1)*n/2,1);

count = 1;
for i = 2:n
    tmp = A(i,1:i-1);
    a(count:count+i-2) = tmp';
    count = count + i -1;
end  