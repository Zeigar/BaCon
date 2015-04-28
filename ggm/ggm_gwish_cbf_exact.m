function [G, K] = ggm_gwish_cbf_exact(G0,S,n,gwishfunc)
% MCMC-Metropolis sampler for Gaussian graphical models with G-Wishart
% prior. The algorithm makes use of the conditional Bayes factor for GGMs
% as described in [1,2], as well as the exact formulation of the partition
% function ratio as described in [3]. A MEX implementation is available for
% speed.
%
% Mandatory parameters: 
%       G: Initialization, could be all zeros or previous sample.
%       S: Zero-mean empirical covariance
%       n: Number of data samples
%
% Optional parameters:
%       gwishfunc: The function for sampling K ~ GWish(.). Default:
%       gwishrnd_mex.
%
% References:
%
% [1] Cheng, Y., & Lenkoski, A. (2012). Hierarchical Gaussian graphical 
% models: Beyond reversible jump. Electronic Journal of Statistics, 6, 
% pp. 2309–2331.
% [2] Hinne, M., Lenkoski, A., Heskes, T., & van Gerven, M. (2014). 
% Efficient sampling of Gaussian graphical models using conditional Bayes 
% factors. Stat 3, pp. 326-336.
% [3] Uhler, C., Lenkoski, A., & Richards, D. (2015). Exact formulas for 
% the normalizing constants of Wishart distributions for graphical models. 
% arXiv.
%
% Last modified: April 28th, 2015
% Max Hinne


if nargin < 4
    gwishfunc = @gwishrnd3;
end


p = length(S);
d = 3;
D = eye(p);
U = D + S;
Uinv = inv(U);

G = max(G0,eye(p));

% draw K from posterior, using current G

a = 1; b = 1; 


linidx = find(triu(ones(p),1));
E = length(linidx);
for e=linidx(randperm(E))'
    % create order with edge e at (p-1,p)
    K = gwishfunc(G, Uinv, d+n, true); while ~ispd(K), K = gwishfunc(G, Uinv, d+n, true);  end;
    [l, m] = ind2sub([p p], e);    
    perm = 1:p; perm([l,m]) = []; perm = [perm(randperm(p-2)), l, m];
    
    % permute all variables of interest
    Gs = G(perm,perm);
    Ks = K(perm,perm);
    Us = U(perm,perm);
        
    Gp = Gs;
    Gp(p-1,p) = 1 - Gp(p-1,p);
    Gp(p,p-1) = 1 - Gp(p,p-1);
    
    if Gp(p-1,p)
        tri = sum(G(p-1,:) .* G(p,:));
        alpha = logncbf(Ks,Us) - log(2*sqrt(pi)) + gammaln( (d + tri)/2 ) - gammaln( (d + tri + 1)/2 );
    else
        tri = sum(Gp(p-1,:) .* Gp(p,:));
        alpha = -logncbf(Ks,Us) + log(2*sqrt(pi)) - gammaln( (d + tri)/2 ) + gammaln( (d + tri + 1)/2 );
    end
    
    logprior = (2 * Gp(p-1,p) - 1) * log(a /  b);
        
    % set G = G' with probability alpha
    if rand < min(1,exp(alpha + logprior))
        Gs = Gp;
    end
    
    % c - unpermute 
    G(perm,perm) = Gs;
end
K = gwishrnd(G, Uinv, d+n, true);

function N = logncbf(K, S)
R = chol(K);
p = length(S);
mu = R(p-1,p-1) * S(p-1,p) / S(p,p);
R0 = -1 / R(p-1,p-1) * sum(R(1:p-2,p-1) .* R(1:p-2,p));
N = log(R(p-1,p-1)) + 0.5*log(2*pi / S(p,p)) + 0.5*S(p,p)*(R0 + mu)^2;
