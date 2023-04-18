function [mu, sigma, p ] = fitGMM(X, K, maxIter)
% Input arguments:
% X: N-by-D data matrix (N data points, D dimensions)
% K: number of Gaussian components
% maxIter: maximum number of iterations
% tol: tolerance for convergence (default: 1e-6)

% Output arguments:
% mu: K-by-D matrix containing the mean vectors of the Gaussian components
% sigma: D-by-D-by-K matrix containing the covariance matrices of the Gaussian components
% p: 1-by-K vector containing the mixing proportions of the Gaussian components
% log_likelihood: 1-by-maxIter vector containing the log-likelihood at each iteration


% Initialize mu, sigma, and p
[N, D] = size(X);
mu = X(randperm(N, K), :);
sigma = repmat(eye(D), [1, 1, K]);
p = ones(1, K) / K;

% Initialize log_likelihood
log_likelihood = zeros(1, maxIter);

%prev_sigma = sigma;
% EM algorithm for GMM
for iter = 1:maxIter
    prev_sigma = sigma;
    disp(iter)
    % E-step: Compute responsibilities
    if iter == 1
        gamma = zeros(N, K);
    end

    for k = 1:K
        diff = abs(X - mu(k, :));
        sigma_k = squeeze(sigma(:, :, k));
        coef = (2*pi)^(D/2) * sqrt(det(sigma_k));
        gamma(:, k) = p(k) * exp(-0.5 * sum((diff * sigma_k) .* diff, 2)) / coef;
    end
    gamma = gamma ./ sum(gamma, 2);
   
    % M-step: Update parameters
    Nk = sum(gamma, 1);
    mu = (gamma' * X) ./ Nk';
    for k = 1:K
        diff = abs(X - mu(k, :));
        sigma_k = (diff'* (diff .* gamma(:, k))) / Nk(k);
        sigma(:, :, k) = sigma_k;
        p(k) = Nk(k) / N;
    end
    
    %Calculating log liklihood
    for k = 1:K
        log_likelihood(iter) = log_likelihood(iter) + log(sum(gamma(:, k)));
    end

    % Check for convergence
    if sum(sum(abs(prev_sigma - sigma))) < 1e-6
        break;
    end
    
end

end