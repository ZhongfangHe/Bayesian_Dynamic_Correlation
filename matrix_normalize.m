% normalize a covariance matrix into a correlation matrix

function R_t = matrix_normalize(Q_t)

tmp = diag(1./sqrt(diag(Q_t)));
R_t = tmp * Q_t * tmp;
