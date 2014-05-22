function [sample] = struct_conn_edge_prior(A0, N, structparam, M, T)
% MCMC-Metropolis sampler for structural connectivity based on a Dirichlet
% compund multinomial distribution, as described in [1], together with a
% fixed prior on the probability of an edge. 
%
% Mandatory parameters: 
%       A0: previous state of connectivity
%       N: streamline count matrix (N_ij = count from i to j)
%
% Optional parameters:
%       structparam.ap, .an: Dirichlet distribution hyperparameters
%       corresponding to a true or a false edge, respectively.
%       M: edge probabilities (assumed symmetric)
%       T: temperature at which to sample (T=1: true distribution). Used in
%       simulated annealing.
%
% References:
%
% [1] Hinne, M., Heskes, T., Beckmann, C.F., & van Gerven, M.AJ. (2013).
% Bayesian inference of structural brain networks. NeuroImage, 66C,
% 543–552.
%
% % Last modified: April 8th, 2014


if nargin <= 2 || isempty(structparam)
    ap = 1;
    an = 0.1;
else
    ap = structparam.ap;
    an = structparam.an;
end

if nargin <= 3 || isempty(M)
    M = 0.5*ones(size(N));
end

if nargin <= 4 || isempty(T)
    T = 1;
end

A = A0;
P = 0;
n = length(A);

linidx = find(triu(ones(n),1));
E = length(linidx);
for e=linidx(randperm(E))'
    Aprop = A;
    [i, j] = ind2sub([n n], e);
    Aprop(i,j) = 1 - A(i,j);
    Aprop(j,i) = 1 - A(j,i);

    dL = delta_log_dcm(N, ap, an, i, j, Aprop, A);    
    q = M(i,j);
    dP = (1 - 2 * Aprop(i,j)) * log( (q-1) / q);

    alpha = dL + dP;
    
    if rand <= min(1,exp(alpha)^(1/T))
        A = Aprop;
        P = P + alpha;    
    end     
end

sample.A = A;
sample.P = P;