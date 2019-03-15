

function A = corr_vec2mat(a, n)
% Inputs:
%    a: a (n-1)*n/2-by-1 vector stacking by rows of A, i.e. a_11, a_21, a_22, a_31, a_32, a_33, ....
%    n: a scalar of the dimension

% Outputs:
%    A: a n-by-n correlation matrix

A = ones(n,n);

count = 1;
for i = 2:n
    tmp = a(count:count+i-2)';
    A(i,1:i-1) = tmp;
    count = count + i -1;
end  
for i = 1:n-1
    A(i,i+1:n) = A(i+1:n,i)';
end