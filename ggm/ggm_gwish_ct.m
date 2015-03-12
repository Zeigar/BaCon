function [G, K, w] = ggm_gwish_ct(G0,S,n)
% for now, assume flat prior on G

p = length(S);
d = 3;
D = eye(p);
U = D + S;
Dinv = inv(D);
Uinv = inv(U);

G = max(G0,eye(p));
K = gwishrnd(G, Uinv, d+n, true); while ~ispd(K), K = gwishrnd(G, Uinv, d+n, true);  end;

linidx = find(triu(ones(p),1));
E = length(linidx);
bd = zeros(E,1);  

for i=1:E
    e = linidx(i);
    if G(e) % death
        [l, m] = ind2sub([p p], e);
        s = 1:p; s([l,m]) = []; s = [s(randperm(p-2)), l, m];

        % permute all variables of interest        
        Gs = G(s,s); Ks = K(s,s); Us = U(s,s); Ds = D(s,s); Dinvs = Dinv(s,s);
        Gp = Gs;
        Gp(p-1,p) = 1 - Gp(p-1,p);
        Gp(p,p-1) = 1 - Gp(p,p-1);

        K0s = gwishrnd(Gp, Dinvs, d, true); while ~ispd(K0s), K0s = gwishrnd(Gp, Dinvs, d, true);  end;
        bd(i) = logncbf(K0s, Ds) - logncbf(Ks, Us);        
    else % birth
        [l, m] = ind2sub([p p], e);
        s = 1:p; s([l,m]) = []; s = [s(randperm(p-2)), l, m];

        % permute all variables of interest  
        Gs = G(s,s); Ks = K(s,s); Us = U(s,s); Ds = D(s,s); Dinvs = Dinv(s,s);
        Gp = Gs;
        Gp(p-1,p) = 1 - Gp(p-1,p);
        Gp(p,p-1) = 1 - Gp(p,p-1);

        K0s = gwishrnd(Gp, Dinvs, d, true); while ~ispd(K0s), K0s = gwishrnd(Gp, Dinvs, d, true);  end;
        bd(i) = logncbf(Ks, Us) - logncbf(K0s, Ds);
    end
end

w_sum_log = logsumexp(bd);
w = -w_sum_log;
% 1.4 simulate jump
    
max_bd = max(bd);
ratios = exp(bd - max_bd) / sum(exp(bd - max_bd));

if sum(ratios)==0
    flipedge = randsample(linidx, 1, true);
else
    flipedge = randsample(linidx, 1, true, ratios);
end

[l, m] = ind2sub([p p], flipedge);
G(l,m) = 1 - G(l,m);
G(m,l) = 1 - G(m,l);

% 2. K' = W_G'(d,D)
K = gwishrnd(G, Uinv, d+n, true); while ~ispd(K), K = gwishrnd(G, Uinv, d+n, true);  end;



function N = logncbf(K, S)
R = chol(K);
p = length(S);
mu = R(p-1,p-1) * S(p-1,p) / S(p,p);
R0 = -1 / R(p-1,p-1) * sum(R(1:p-2,p-1) .* R(1:p-2,p));
N = log(R(p-1,p-1)) + 0.5*log((2*pi / S(p,p))) + 0.5*S(p,p)*(R0 + mu)^2;