% Q_t from VC (set initial values Q_1 to Q_M = E(u2))

function q_mat = VC_filter(u2, M, MA_mat)
% Inputs:
%   u2: a T-by-((n+1)*n/2) matrix of vectorized u_t * u_t'.
%   M: a scalar of the length of moving window.
%   MA_mat: a (T-M)*T matrix with diagonal ones(1,M).
% Outputs:
%   q_mat: a T-by-((n+1)*n/2) matrix of vectorized Q_t.


mean_u2 = mean(u2);


%% Q_1 to Q_M are mean(u2)
q_mat1 = kron(ones(M,1), mean_u2);


%% Q_{M+1} to Q_T are moving average of u2 with the window length of M
q_mat2 = MA_mat * u2 / M; 

%% Stack Q mat
q_mat = [q_mat1; q_mat2];