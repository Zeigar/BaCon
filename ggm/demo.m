% addpath ~/Code/L1General/misc/;
% addpath ~/Code/BCToolbox/utility/;

%% setup

p = 6; n = 18;
A = eye(p);
A(2:p+1:end) = 0.5;
A(p+1:p+1:end) = 0.5;
A(1,p) = 0.4; A(p,1) = 0.4;
S = n*inv(A);

WL_G = [1     0.969   0.106   0.085   0.113   0.85;
        0.969 1       0.98    0.098   0.081   0.115;
        0.106 0.98    1       0.982   0.0098  0.086;
        0.085 0.098   0.982   1       0.98    0.106;
        0.113 0.081   0.098   0.98    1       0.97;
        0.85  0.115   0.086   0.106   0.97    1];

WL_K = [1.139     0.569   -0.011   0.006   -0.013   0.403;
        0.569 1.175       0.574    -0.008   0.005   -0.014;
        -0.011 0.574    1.176       0.574   -0.008  0.006;
        0.006 -0.008   0.574   1.175       0.573    -0.011;
        -0.013 0.005   -0.008   0.573    1.175       0.569;
        0.403  -0.014   0.006   -0.011   0.569    1.138];
    
G0 = A>0;

%% simulation
sim=tic;
iters=1e5; burn = iters/2;

% Double reversible-jump
fprintf('Double reversible-jump\n');
DRJ_Gs = zeros(p,p,iters);
DRJ_Ks = zeros(p,p,iters);
G = eye(p);
DRJ_start=tic;
for i=1:iters
    if mod(i,iters/10)==0
        fprintf('iter %d\n', i)
    end
    [G, K] = ggm_gwish_drj(G,S,n);
    DRJ_Gs(:,:,i) = G;
    DRJ_Ks(:,:,i) = K;
end
s = toc(DRJ_start);
DRJ_SpS = iters/s;


% Double conditional Bayes factor
fprintf('Double conditional Bayes factor\n');

DCBF_Gs = zeros(p,p,iters);
DCBF_Ks = zeros(p,p,iters);
G = eye(p);
DCBF_start=tic;
for i=1:iters
    if mod(i,iters/10)==0
        fprintf('iter %d\n', i)
    end
    [G, K] = ggm_gwish_cbf_direct(G,S,n);
    DCBF_Gs(:,:,i) = G;
    DCBF_Ks(:,:,i) = K;
end
s = toc(DCBF_start);
DCBF_SpS = iters/s;

% continuous-time double conditional Bayes factor MCMC
fprintf('Continuous-time double conditional Bayes factor\n');
CT_Gs = zeros(p,p,iters+1);
CT_Ks = zeros(p,p,iters+1);
CT_ws = zeros(iters+1,1);
G = eye(p);
CT_start=tic;
for i=1:iters+1 % because we obtain [G', K', w] 
    if mod(i,iters/10)==0
        fprintf('iter %d\n', i)
    end
    [G, K, w] = ggm_gwish_ct(G,S,n);
    CT_Gs(:,:,i) = G;
    CT_Ks(:,:,i) = K;
    CT_ws(i) = w;
end
s=toc(CT_start);
CT_SpS = iters/s;
toc(sim);

%% analysis

MSE = zeros(3,1);
KL = zeros(3,1);
Pmode = zeros(3,1);
Nmodels = zeros(3,1);

DRJ_G = squeeze(mean(DRJ_Gs(:,:,burn+1:end),3));
DRJ_K = squeeze(mean(DRJ_Ks(:,:,burn+1:end),3));
    
DCBF_G = squeeze(mean(DCBF_Gs(:,:,burn+1:end),3));
DCBF_K = squeeze(mean(DCBF_Ks(:,:,burn+1:end),3));

DCT_G = weighted_exp(CT_Gs(:,:,burn+2:end-1), CT_ws(burn+1:end-1));
DCT_K = weighted_exp(CT_Ks(:,:,burn+2:end-1), CT_ws(burn+1:end-1));    

MSE(1) = mean(power(DRJ_G(find(triu(ones(p),1))) - WL_G(find(triu(ones(p),1))), 2));
MSE(2) = mean(power(DCBF_G(find(triu(ones(p),1))) - WL_G(find(triu(ones(p),1))), 2));
MSE(3) = mean(power(DCT_G(find(triu(ones(p),1))) - WL_G(find(triu(ones(p),1))), 2));
    
KL(1) = KLdiv(WL_K,DRJ_K);
KL(2) = KLdiv(WL_K,DCBF_K);
KL(3) = KLdiv(WL_K,DCT_K);
    
[pr, ix] = sample_dist(DRJ_Gs(:,:,burn+1:end));
[prsort, ixsort] = sort(pr, 'descend');
Nmodels(1) = length(pr);

for i=1:length(pr)
    if all(all(G0==DRJ_Gs(:,:,burn+ix(ixsort(i)))))
        Pmode(1) = prsort(i);
        break; 
    end
end
    
[pr, ix] = sample_dist(DCBF_Gs(:,:,burn+1:end));
[prsort, ixsort] = sort(pr, 'descend');
Nmodels(2) = length(pr);

for i=1:length(pr)
    if all(all(G0==DCBF_Gs(:,:,burn+ix(ixsort(i)))))
        Pmode(2) = prsort(i);
        break; 
    end
end
    
[pr, ix] = sample_dist(CT_Gs(:,:,burn+2:end), CT_ws(burn+1:end-1));
[prsort, ixsort] = sort(pr, 'descend');
Nmodels(3) = length(pr);

for i=1:length(pr)
    if all(all(G0==CT_Gs(:,:,burn+ix(ixsort(i)))))
        Pmode(3) = prsort(i);
        break; 
    end
end

fprintf('MSE: %.4f, %.4f, %.4f\n', MSE);
fprintf('KL: %.4f, %.4f, %.4f\n', KL);
fprintf('P(mode): %.4f, %.4f, %.4f\n', Pmode);% NB short sampling may not encounter the mode, hence P(mode) = 0
fprintf('#models: %d, %d, %d\n', Nmodels);