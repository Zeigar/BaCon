clear all;
cd ggm\;
load ../demodata.mat;

%% estimate P(G,K|X)
% Posterior of conditional independence and partial correlations

[n,p] = size(subcort);

Z = zscore(subcort);
S = Z'*Z;
G = eye(p);
nsamples = 1e3;

fprintf('Loaded data, p=%d, N=%d\n', p, n);

fprintf('Drawing %d samples from P(G,K|X) using Matlab only... ', nsamples);

Gsamples_mlb = zeros(p,p,nsamples);
Ksamples_mlb = zeros(p,p,nsamples);
tic;
for i=1:nsamples
    [G,K] = ggm_gwish_cbf_direct(G,S,n,@gwishrnd3);    
    Gsamples_mlb(:,:,i) = G;
    Ksamples_mlb(:,:,i) = K;
end
tmatlab = toc;
mean(Gsamples_mlb,3);
fprintf(' done in %4.2f seconds.\n', tmatlab);

fprintf('Drawing %d samples from P(G,K|X) using Matlab + MEX file for G-Wishard samples... ', nsamples);

Gsamples_mex = zeros(p,p,nsamples);
Ksamples_mex = zeros(p,p,nsamples);
tic;
for i=1:nsamples
    [G,K] = ggm_gwish_cbf_direct(G,S,n,@gwishrnd_mex);    
    Gsamples_mex(:,:,i) = G;
    Ksamples_mex(:,:,i) = K;
end
tmexgwish = toc;
mean(Gsamples_mex,3);
fprintf(' done in %4.2f seconds.\n', tmexgwish);

fprintf('Drawing %d samples from P(G,K|X) using MEX routine... ', nsamples);
tic; 
[Gsamples,Ksamples] = ggm_cbf_mex(G,S,n,nsamples); 
tmexggm = toc;
mean(Gsamples,3);
fprintf(' done in %4.2f seconds.\n', tmexggm);

fprintf('%12s | %12s | %12s | %12s \n', 'Method', 'Matlab', 'MEX gwishrnd', 'MEX GGM');
fprintf('%12s | %11.2fs | %11.2fs | %11.2fs\n', 'Time', tmatlab, tmexgwish, tmexggm);