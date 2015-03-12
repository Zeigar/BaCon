function [G,K,ar] = ggm_gwish_drj(G0,U,n)
% Implements a MCMC-Metropolis sampler for structural connectivity G and 
% functional connectivity K [1]. 
%
% Input: 
%   required    - X (n x p): timeseries
%   "           - G0 (p x p): initialization of G
%   optional    - funcparam: 
%                   - delta: prior d-o-f
%                   - D: prior scatter matrix
%                   - sig2_g: jump variance for elements of K
%               - priorparam:
%                   - alpha/beta: Beta hyperparameter governing graph
%                   density
%
% Literature
% [1] Alex Lenkoski, 2013. A direct sampler for G-Wishart variates. Stat,
%   2(1), pp. 119-–128.


p = length(G0);

% flat prior
alpha = 1;
beta = 1;



delta = 3;
D = eye(p);
sig2_g = 1;


% U = X'*X;

% pre-compute inverses
DUinv = inv(D+U);
Dinv = inv(D);

G = max(G0,eye(p)); 

acc_add = 0;
adds=0;
acc_rem = 0;
rems=0;
pdretries = 0;

linidx = find(triu(ones(p),1));
E = length(linidx);
for e=linidx(randperm(E))'
    Gp = G;
    K = gwishrnd(G, DUinv, delta+n, true); while ~ispd(K), pdretries = pdretries+1; K = gwishrnd(G, DUinv, delta+n, true);  end
    R = chol(K);
    [l, m] = ind2sub([p p], e);
    Gp(l,m) = 1 - G(l,m);
    Gp(m,l) = 1 - G(m,l);
    
    if Gp(l,m) % additional edge
        % first jump, in posterior
        R = chol(K);
        
        gamma = normrnd(R(l,m), sig2_g); 
        
        Rprop = zeros(p);
        Gix = find( triu(G) );    
        Rprop(Gix) = R(Gix);
        Rprop(l,m) = gamma;    
        Rprop = complete(Rprop, Gp);
        Kprop = Rprop' * Rprop;
        
        % second jump, in prior        
        K0prop = gwishrnd(Gp, Dinv, delta, true); while ~ispd(K0prop), pdretries = pdretries+1; K0prop = gwishrnd(Gp, Dinv, delta, true);  end
        R0prop = chol(K0prop);
        
        R0 = zeros(p);
        Gix = find( triu(G) );
        R0(Gix) = R0prop(Gix);
        R0 = complete(R0, G);
        K0 = R0' * R0;

        logfunc = log( R(l,l) / R0(l,l) ) ... 
            - 0.5 * trace((Kprop - K)' * (D + U)) - 0.5 * trace((K0 - K0prop)' * D) ...
            + ((Rprop(l,m) - R(l,m))^2 - (R0prop(l,m) - R0(l,m))^2) / (2*sig2_g);        
        adds=adds+1;
    else % edge removal, see [3] 
        % first jump, in posterior
        
        Rprop = zeros(p);
        Gix = find( triu(Gp) ); 
        Rprop(Gix) = R(Gix);
        Rprop = complete(Rprop, Gp); 
        Kprop = Rprop'*Rprop;
             
        % second jump, in prior                
        K0prop = gwishrnd(Gp, Dinv, delta, true); while ~ispd(K0prop), pdretries = pdretries+1; K0prop = gwishrnd(Gp, Dinv, delta, true); end
        R0prop = chol(K0prop);
        
        gammaprop = normrnd(R0prop(l,m), sig2_g);

        R0 = zeros(p);
        Gix = find( triu(Gp) );
        R0(Gix) = R0prop(Gix); 
        R0(l,m) = gammaprop; 
        R0 = complete(R0, G);
        K0 = R0'*R0;
        
        logfunc = log( R0(l,l) / R(l,l) ) ... 
            - 0.5 * trace((Kprop - K)' * (D + U)) - 0.5 * trace((K0 - K0prop)' * D) ...
            - ((Rprop(l,m) - R(l,m))^2 - (R0prop(l,m) - R0(l,m))^2) / (2*sig2_g);
        rems=rems+1;
    end
    
    logprior = (2 * Gp(l,m) - 1) * log(alpha /  beta);
    
    if rand < min(1, exp(logfunc + logprior)) % accept
        if Gp(l,m)
            acc_add = acc_add+1;
        else
            acc_rem = acc_rem+1;
        end
        G = Gp;        
    end
end

ar = (acc_add+acc_rem) / E;
G = sparse(G);
K = sparse(gwishrnd(G, DUinv, delta + n, true));

function R = complete(R,G)
[I, J] = find(triu(~G));
E = length(I);
for e=1:E
    i = I(e);
    j = J(e);        
    R(i,j) = -(R(1:i,i)' * R(1:i,j)) / R(i,i);
end