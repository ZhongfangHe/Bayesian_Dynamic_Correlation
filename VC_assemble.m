% Assemble R_t from X_t, where X_t = normalize(Q_t) (columns of ones are removed)
% R_t = (1-a-b)*R + a*X_t + b*R_{t-1}
% where R_1 = R = normalize(E(u2))

function r_mat = VC_assemble(x_mat, a, b, R0)
% Inputs:
%   x_mat: a T-by-((n-1)*n/2) matrix of vectorized correlation matrices (columns of ones are removed)
%   a: a scalar of coef on X_t
%   b: a scalar of coef on R_{t-1}
%   R0: a n-by-n matrix of the normalized E(u_t * u_t')
% Outputs:
%   r_mat: a T-by-((n-1)*n/2) matrix of vectorized correlation matrices (columns of ones are removed)

T = size(x_mat,1);

r0_vec = corr_mat2vec(R0); %initial value
const_vec = (1-a-b)*r0_vec; %constant = (1-a-b)*R0


b_vec = b.^(0:T)';
coef_vec_const = (1 - b_vec(2:(T+1))) / (1-b); %coef on the constant
coef_vec_r0 = b_vec(2:(T+1)); %coef on the initial value R0
coef_mat_x = eye(T);
for i = 1:(T-1)
    coef_mat_x((i+1):T,i) = b_vec(2:(T-i+1));
end %coef on X_t

r_mat = coef_vec_const * const_vec' + a * coef_mat_x * x_mat + coef_vec_r0 * r0_vec';







