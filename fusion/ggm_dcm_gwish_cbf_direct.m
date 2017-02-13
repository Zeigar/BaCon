function [G, K, ar] = ggm_dcm_gwish_cbf_direct(G0,S,n,N,gwishfunc,J)

if nargin < 5 || isempty(gwishfunc)
    gwishfunc = @gwishrnd3;
end

if nargin < 6 || isempty(J)
    J = 0;
end

p = length(S);
d = 3;
D = eye(p);
Dinv = inv(D);
U = D+S;
Uinv = inv(U);

G = max(G0,eye(p));

linidx = find(triu(ones(p),1));
E = length(linidx);

a = 1; b = 1; 
ap = 1; an = 0.1;

accepts = 0;

for e=linidx(randperm(E))'
    % draw K from posterior, using current G
    K = gwishfunc(G, Uinv, d+n, true); while ~ispd(K), K = gwishfunc(G, Uinv, d+n, true);  end;
    
    % create order with edge e at (p-1,p)
    [l, m] = ind2sub([p p], e);
    perm = 1:p; perm([l,m]) = []; perm = [perm(randperm(p-2)), l, m];
    
    % permute all variables of interest
    Gs = G(perm,perm);
    Ks = K(perm,perm);
    Us = U(perm,perm);
    Ds = D(perm,perm);
    Dinvs = Dinv(perm,perm);
    
    % obtain cholesky decomp. of permuted K
    Rs = chol(Ks);
    
    Gp = Gs;
    Gp(p-1,p) = 1 - Gp(p-1,p);
    Gp(p,p-1) = 1 - Gp(p,p-1);

    % K from prior
    K0s = gwishfunc(Gp, Dinvs, d, true); while ~ispd(K0s), K0s = gwishfunc(Gp, Dinvs, d, true);  end;
    R0s = chol(K0s);
    
    if Gp(p-1,p) % additional edge        
        logfunc = logncbf(Rs, Us) - logncbf(R0s, Ds);    
    else % edge removal, see [3] 
        logfunc = logncbf(R0s, Ds) - logncbf(Rs, Us);        
    end 
    
    logstruct = DCM_DL(N(perm,perm), ap, an, p-1, p, Gp, Gs);
    logprior = (2 * Gp(p-1,p) - 1) * log(a /  b);
    
    alpha = logfunc + logstruct + logprior;
        
    % set G = G' with probability alpha
    if rand < min(1,exp(alpha))
        accepts = accepts + 1;
        Gs = Gp;
    end
    
    % c - unpermute 
    G(perm,perm) = Gs;
    K(perm,perm) = Ks;
end

ar = accepts / E;

GJ = G;
% big jump through consecutive steps
if J > 1
%     fprintf('attempting jump\n');
    alpha = 0;
%     states = zeros(p,p,2*J);
    for e=randsample(linidx,2*J)'
        % draw K from posterior, using current G
        KJ = gwishfunc(GJ, Uinv, d+n, true); while ~ispd(KJ), KJ = gwishfunc(GJ, Uinv, d+n, true);  end;

        % create order with edge e at (p-1,p)
        [l, m] = ind2sub([p p], e);
        perm = 1:p; perm([l,m]) = []; perm = [perm(randperm(p-2)), l, m];

        % permute all variables of interest
        Gs = GJ(perm,perm);
        Ks = KJ(perm,perm);
        Us = U(perm,perm);
        Ds = D(perm,perm);
        Dinvs = Dinv(perm,perm);

        % obtain cholesky decomp. of permuted K
        Rs = chol(Ks);

        Gp = Gs;
        Gp(p-1,p) = 1 - Gp(p-1,p);
        Gp(p,p-1) = 1 - Gp(p,p-1);

        % K from prior
        K0s = gwishfunc(Gp, Dinvs, d, true); while ~ispd(K0s), K0s = gwishfunc(Gp, Dinvs, d, true);  end;
        R0s = chol(K0s);

        if Gp(p-1,p) % additional edge        
            logfunc = logncbf(Rs, Us) - logncbf(R0s, Ds);    
        else % edge removal, see [3] 
            logfunc = logncbf(R0s, Ds) - logncbf(Rs, Us);        
        end 

        logstruct = DCM_DL(N(perm,perm), ap, an, p-1, p, Gp, Gs);
        logprior = (2 * Gp(p-1,p) - 1) * log(a /  b);

        alpha = alpha + logfunc + logstruct + logprior;
        
        Gs = Gp;
        % c - unpermute 
        GJ(perm,perm) = Gs;
        KJ(perm,perm) = Ks;
    end
    acceptJ = false;
    % set G = G' with probability alpha
    if rand < min(1,exp(alpha))
        acceptJ = true;
        G = GJ;
        K = KJ;
    end   
    if acceptJ
        fprintf('jump accepted\n');
%     else
%         fprintf('jump rejected\n')
    end
end


function N = logncbf(R, S)
p = length(S);
mu = R(p-1,p-1) * S(p-1,p) / S(p,p);
R0 = -1 / R(p-1,p-1) * sum(R(1:p-2,p-1) .* R(1:p-2,p));
N = log(R(p-1,p-1)) + 0.5*log((2*pi / S(p,p))) + 0.5*S(p,p)*(R0 + mu)^2;
