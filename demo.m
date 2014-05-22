%% Bayesian Connectomics DEMO

clear all;
load demodata.mat;
[n,p] = size(X);

% estimate structural connectivity

prior.a = 1;
prior.b = 4; % expected density = a/(a+b)

% full distribution

nsamples = 100;
structuralsamples = zeros(p,p,nsamples);
A = zeros(p);

tic;
for i=1:nsamples
    sample = struct_conn_density_prior(A,N,[],prior);
    A = sample.A;
    structuralsamples(:,:,i) = A;
end
toc;

A_expectation = squeeze(mean(structuralsamples,3));

figure;
imagesc(A_expectation);
colormap hot;
axis square; colorbar;

% MAP estimate

nsamples = 100;
A_map = zeros(p);
T = 10;
Tr = .5^(10/nsamples);

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

% full distribution with edge-wise prior

nsamples = 100;
structuralsamples2 = zeros(p,p,nsamples);
A2 = zeros(p);

% Here we specify a silly prior that favors intra-hemisphere connections
M = zeros(p);
M(1:34,1:34)   = 1; 
M(35:68,35:68) = 1; 
M(1:34,69:75)  = 1; 
M(69:75,1:34)  = 1; 
M(35:68,76:82) = 1; 
M(76:82,35:68) = 1; 
M(69:75,69:75) = 1; 
M(76:82,76:82) = 1;
M = 0.9*M + 0.1*~M;

tic;
for i=1:nsamples
    sample = struct_conn_edge_prior(A2,N,[],M);
    A2 = sample.A;
    structuralsamples2(:,:,i) = A2;
end
toc;

A2_expectation = squeeze(mean(structuralsamples2,3));

figure;
subplot 121;
imagesc(M);
colormap hot;
axis square; colorbar;
subplot 122;
imagesc(A2_expectation);
colormap hot;
axis square; colorbar;


% estimate functional connectivity with structural constraint

nsamples = 1000;
functionalsamples = zeros(p,p,nsamples);
S = cov(X);
df = 3 + n;

tic;
for i=1:nsamples
    K = gwishrnd(A_map + eye(p), S, df); % diagonal must be ones
    functionalsamples(:,:,i) = prec2parcor(K);    
end
toc;

R_expectation = squeeze(mean(functionalsamples,3));

figure;
imagesc(R_expectation.*(A_map+eye(p)));
colormap jet;
axis square; colorbar;
caxis([-1 1]);
