function [G, K, ar] = ggm_gwish_cbf_direct(G0,S,n,gwishfunc)

if nargin < 4
    gwishfunc = @gwishrnd3;
end


p = length(S);
d = 3;
D = eye(p);
Dinv = inv(D);
U = D + S;
Uinv = inv(U);

G = max(G0,eye(p));

% draw K from posterior, using current G

a = 1; b = 1; 

accepts = 0;

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
    Ds = D(perm,perm);
    Dinvs = Dinv(perm,perm);
        
    Gp = Gs;
    Gp(p-1,p) = 1 - Gp(p-1,p);
    Gp(p,p-1) = 1 - Gp(p,p-1);

    % K from prior data and proposal graph
    K0s = gwishfunc(Gp, Dinvs, d, true); while ~ispd(K0s), K0s = gwishfunc(Gp, Dinvs, d, true);  end;
    
    if Gp(p-1,p) % additional edge        
        alpha = logncbf(Ks, Us) - logncbf(K0s, Ds);    
    else % edge removal
        alpha = logncbf(K0s, Ds) - logncbf(Ks, Us);        
    end 
    
    logprior = (2 * Gp(p-1,p) - 1) * log(a /  b);
        
    % set G = G' with probability alpha
    if rand < min(1,exp(alpha + logprior))
        accepts = accepts + 1;
        Gs = Gp;
    end
    
    % c - unpermute 
    G(perm,perm) = Gs;
    K(perm,perm) = Ks;
end
%K = gwishrnd(G, Uinv, d+n, true);
ar = accepts / E;

function N = logncbf(K, S)
R = chol(K);
p = length(S);
mu = R(p-1,p-1) * S(p-1,p) / S(p,p);
R0 = -1 / R(p-1,p-1) * sum(R(1:p-2,p-1) .* R(1:p-2,p));
N = log(R(p-1,p-1)) + 0.5*log(2*pi / S(p,p)) + 0.5*S(p,p)*(R0 + mu)^2;
