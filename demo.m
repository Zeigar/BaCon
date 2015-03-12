%% Bayesian Connectomics DEMO
addpath utility\;
addpath ggm\;
clear all;
load demodata.mat;

%% estimate P(G,K|X)
% Posterior of conditional independence and partial correlations

[n,p] = size(subcort);

Z = zscore(subcort);
S = Z'*Z;

nsamples = 1000;

G = eye(p); % initial sample
Gsamples = zeros(p,p,nsamples); % conditional independencies
Ksamples = zeros(p,p,nsamples); % precision matrices
Rsamples = zeros(p,p,nsamples); % partial correlations

for i=1:nsamples
    [G,K] = ggm_gwish_cbf_direct(G,S,n);
    Gsamples(:,:,i) = G;
    Ksamples(:,:,i) = K;
    Rsamples(:,:,i) = prec2parcor(K);
end

figure; imagesc(mean(Gsamples,3)); axis square; colormap hot; caxis([0 1]);
figure; imagesc(mean(Rsamples,3)); axis square; colormap jet; caxis([-1 1]);

%% estimate structural connectivity

p = length(N);
prior.a = 1;
prior.b = 4; % expected density = a/(a+b)

%% full distribution

nsamples = 5000;
structuralsamples = zeros(p,p,nsamples);
A = zeros(p);

tic;
for i=1:nsamples
    sample = struct_conn_density_prior(A,N);
    A = sample.A;
    structuralsamples(:,:,i) = A;
end
toc;

A_expectation = squeeze(mean(structuralsamples,3));

figure;
imagesc(A_expectation);
colormap hot;
axis square; colorbar;

%% MAP estimate

nsamples = 100;
A_map = zeros(p);
T = 10; % initial temperature
Tr = .5^(10/nsamples); % temperature decay

tic;
for i=1:nsamples    
    sample_map = struct_conn_density_prior(A_map,N,[],prior, T);
    A_map = sample.A;
    T = T*Tr;
end
toc;

figure;
imagesc(A_map);
colormap hot;
axis square; colorbar;


%% estimate functional connectivity with structural constraint

nsamples = 1000;
Rsamples = zeros(p,p,nsamples);
S = cov(X);
df = 3 + n;

tic;
for i=1:nsamples
    K = gwishrnd(A_map + eye(p), S, df); % diagonal must be ones
    Rsamples(:,:,i) = prec2parcor(K);    
end
toc;

R_expectation = squeeze(mean(Rsamples,3));

figure;
imagesc(R_expectation.*(A_map+eye(p)));
colormap jet;
axis square; colorbar;
caxis([-1 1]);

%% estimate MAP structural connectivity with nonparametric clustering prior

mem = crprnd(log(p),p);
Z = mem2cluster(mem);

Ns = {N};
nsubjects = length(Ns);
As = {};

for i=1:nsubjects
    As{i} = zeros(p);
end

nsamples = 1000;

conn_samples = cell(nsamples,nsubjects);
clust_samples = cell(nsamples,1);
num_clusters = zeros(nsamples,1);

T = 10;
Tr = .5^(10/nsamples);

prior.alpha = log(p);
prior.betap = [1 1]; % [1 1] for uninformative prior, [x y], with x>y for modular clusters
prior.betan = fliplr(prior.betap);

for i=1:nsamples    
    [As, Z] = struct_conn_irm_prior(As, Z, N, {}, prior, T);
    for n=1:nsubjects
        conn_samples{i,n} = As{n}+As{n}';
    end
    clust_samples{i} = Z;
    num_clusters(i) = size(Z,1);
    T = T*Tr;
end


figure;
for s=1:nsubjects
    A = conn_samples{end,s};
    Z = clust_samples{end};
    subplot(1,nsubjects,s);
    plot_clustering(A,Z);  
end

figure;
plot(num_clusters);


