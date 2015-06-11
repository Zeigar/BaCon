function [G,P] = struct_conn_density_prior(G0, N, structparam, priorparam, T)
% MCMC-Metropolis sampler for structural connectivity based on a Dirichlet
% compund multinomial distribution, as described in [1], together with a
% stochastic prior on the probability of an edge. 
%
% Mandatory parameters: 
%       A0: previous state of connectivity
%       N: streamline count matrix (N_ij = count from i to j)
%
% Optional parameters:
%       structparam.ap, .an: Dirichlet distribution hyperparameters
%       corresponding to a true or a false edge, respectively.
%       priorparam.a, .b: Beta distribution hyperparameters determining the
%       probability of an edge.
%       T: temperature at which to sample (T=1: true distribution). Used in
%       simulated annealing.
%
% References:
%
% [1] Hinne, M., Heskes, T., Beckmann, C.F., & van Gerven, M.AJ. (2013).
% Bayesian inference of structural brain networks. NeuroImage, 66C,
% 543–552.
%
% Last modified: April 28th, 2015


if nargin <= 3 || isempty(structparam)
    ap = 1;
    an = 0.1;
else
    ap = structparam.ap;
    an = structparam.an;
end

if nargin <= 4 || isempty(priorparam)
    a = 1;
    b = 1;
else
    a = priorparam.a;
    b = priorparam.b;
end

if nargin < 5 || isempty(T)
    T = 1;
end

G = G0;
P = 0;
n = length(G);

linidx = find(triu(ones(n),1));
E = length(linidx);
for e=linidx(randperm(E))'
    Gprop = G;
    [i, j] = ind2sub([n n], e);
    Gprop(i,j) = 1 - G(i,j);
    Gprop(j,i) = 1 - G(j,i);

    dL = delta_log_dcm(N, ap, an, i, j, Gprop, G);    
    dP = (1 - 2 * Gprop(i,j)) * log( a /  b);

    alpha = dL + dP;
    
    if rand <= min(1,exp(alpha)^(1/T))
        G = Gprop;
        P = P + alpha;    
    end     
end
