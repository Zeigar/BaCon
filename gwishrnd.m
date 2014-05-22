function K = gwishrnd(G,S,df,optimize)
% Draw a sample from the G-Wishart distribution [1].
%
% Mandatory parameters: 
%       G: graph structure
%       S: empirical covariance OR inverse of empirical covariance if
%       optimize==true
%       df: degrees of freedom
%
% Optional parameters:
%       optimize: set to true if the inverse of S is precomputed (default:
%       false).
%
% References:
%
% [1] Lenkoski, A. (2013). A direct sampler for G-Wishart variates. Stat,
% 2(1), 119–128.
%
% Last modified: April 8th, 2014

if nargin < 4 || isempty(optimize)
    optimize = false;
end

p       = length(G);

if optimize
    Kp      = wishrnd_opt(S, df+p-1);   % slow
else
    Kp      = wishrnd_opt(inv(S), df+p-1);   % slow
end
% this notation differs from [1], because of different
% parameterizations, e.g. S vs inv(S) and n=df+p-1
Sigma   = inv(Kp);
W       = Sigma;
Wprev   = zeros(p);
tol     = 1e-8;

row = 1:p;
not_j = true(p,1);

i = 0;
while max(abs(Wprev - W)) > tol;
    Wprev = W;
    for j=1:p
        not_j(j) = false;
        a = G(j,:);
        a(j) = 0;
        N_j = row(a==1);
        if ~isempty(N_j)
            W_N_j = W(N_j,N_j);
            Sigma_N_j_j = Sigma(N_j,j);
            bp_j = W_N_j \ Sigma_N_j_j;  % slow
            b_j = zeros(p,1);
            b_j(N_j) = bp_j;
            b_j(j) = [];        
            W_nj_nj = W(not_j,not_j);
            W(j,not_j) = W_nj_nj * b_j;
            W(not_j,j) = W_nj_nj * b_j;
        else
            W(j,not_j) = 0;
            W(not_j,j) = 0;
        end
        not_j(j) = true;
    end
    i = i + 1;
end

K = inv(W) .* max(G,eye(p)); % just to round the almost-zeros to zero