function K = gwishrnd3(G,S,df,optimize)
% Lenkoski, A. (2013). A direct sampler for G-Wishart variates. Stat, 2(1), 119?128. doi:10.1002/sta4.23

if nargin < 4 || isempty(optimize)
    optimize = false;
end

p       = length(G);

if optimize
    Kp      = wishrnd_opt(S, df+p-1);   % slow
else
    Kp      = wishrnd_opt(inv(S), df+p-1);   % slow
end
% this notation differs from Lenkoski's algorithm, because of different
% parameterizations, e.g. S vs inv(S) and n=df+p-1
Sigma   = inv(Kp);
W       = Sigma;
Wprev   = zeros(p);
tol     = 1e-5; % what is a good tolerance setting?

G = logical(min(G, ~eye(p))); % set diagonal to zero so we don't find self-loops

%% precompute stuff

N = cell(1,p);
for j=1:p
    a = G(j,:);
    a(j) = 0;
    N{j} = find(a); %(a==1);
end

neg=cell(1,p);
for j=1:p
%     neg{j} = (1:p~=j);
    neg{j} = [1:j-1, j+1:p];
end

i = 0;
while max(max(abs(Wprev - W))) > tol;
    Wprev = W;
    for j=1:p
         N_j = N{j};
        not_j = neg{j};
        if ~isempty(N_j)  
            R = chol(W(N_j,N_j));
            
            W(not_j,j) = W(not_j,N_j) * (R \ (R'\ Sigma(N_j,j))); 
        else
            W(not_j,j) = 0;
        end
        W(j,not_j) = W(not_j,j);
    end
    i = i + 1;
%     i
end

K = inv(W) .* max(G,eye(p)); % just to round the almost-zeros to zero