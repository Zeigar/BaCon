function [A, Z] = struct_conn_irm_prior(A, Z, Ns, structparam, priorparam, T)

if ~iscell(A)
    B=A;
    clear A;
    A{1}=B;
    clear B;
end

if ~iscell(Ns)
    B=Ns;
    clear Ns;
    Ns{1}=B;
    clear B;
end

N = length(A);
p = length(A{1});

if nargin <= 3 || isempty(structparam)
    ap = 1;
    an = 0.1;
else
    ap = structparam.ap;
    an = structparam.an;
end

if nargin <= 4 || isempty(priorparam)
    alpha = log(p);
    betap = [1 1];
    betan = [1 1];
else
    alpha = priorparam.alpha;
    betap = priorparam.betap;
    betan = priorparam.betan;
end

if nargin < 5 || isempty(T)
    T = 1;
end


for n=1:N    
    A{n}=triu(A{n});    
end


% Gibbs sampling of P(Z|A)
[Z,logP_A,logP_Z] = irm_gibbs(T,Z,A,betap,betan,alpha,randperm(p)); 
Z = irm_split_merge(T, Z,A,betap,betan,alpha,logP_A,logP_Z);

for n=1:N    
    A{n}=triu(A{n});    
end

noc = size(Z,1);

% MH-sample connectivity P(A|Z,N)

for i=1:N
    links = Z*(A{i}+A{i}')*Z';
    links = links - 0.5 * diag(diag(links));  
    Ap = links + betap(1)*eye(noc) + betap(2)*~eye(noc);
    E = ~eye(p);
    nonlinks = Z*E*Z';
    nonlinks = nonlinks - 0.5 * diag(diag(nonlinks)) - links;
    An = nonlinks + betan(1)*eye(noc) + betan(2)*~eye(noc); 
    [A{i}] = struct_conn_metropolis(T, A{i}, Ns{i}, Z, {Ap An}, ap, an);
end


function [Z,logP_A,logP_Z]=irm_split_merge(T,Z,A,eta0p,eta0n,alpha,logP_A,logP_Z)

%[logP_A_t,logP_Z_t]=evalLikelihood(Z,A,W,eta0p,eta0n,alpha,type,method);   
noc=size(Z,1);
J=size(A{1},1);    

% step 1 select two observations i and j        
ind1=ceil(J*rand);        
ind2=ceil((J-1)*rand);
if ind1<=ind2
   ind2=ind2+1;
end
clust1=find(Z(:,ind1));
clust2=find(Z(:,ind2));

if clust1==clust2 % Split  
    setZ=find(sum(Z([clust1 clust2],:)));    
    setZ=setdiff(setZ,[ind1,ind2]);
    n_setZ=length(setZ);
    Z_t=Z;
    Z_t(clust1,:)=0;        
    comp=[clust1 noc+1];               
    Z_t(comp(1),ind1)=1;
    Z_t(comp(2),ind2)=1;

    % Reassign by restricted gibbs sampling        
    if n_setZ>0
        for rep=1:3
            [Z_t,logP_A_t,logP_Z_t,logQ_trans,comp]=irm_gibbs(T,Z_t,A,eta0p,eta0n,alpha,setZ(randperm(n_setZ)),comp);                        
        end     
    else
       logQ_trans=0;
       [logP_A_t,logP_Z_t]=evalLikelihood(Z_t,A,eta0p,eta0n,alpha);                 
    end

    % Calculate Metropolis-Hastings ratio
    a_split=rand<exp((logP_A_t-logP_A + logP_Z_t-logP_Z)/T - logQ_trans); 
    if a_split
       logP_A=logP_A_t;
       logP_Z=logP_Z_t;
       Z=Z_t;
    end
else % Merge                                     
    Z_t=Z;
    Z_t(clust1,:)=Z_t(clust1,:)+Z_t(clust2,:);
    setZ=find(Z_t(clust1,:));           
    Z_t(clust2,:)=[];        
    if clust2<clust1
        clust1_t=clust1-1;
    else 
        clust1_t=clust1;
    end
    noc_t=noc-1;

    % calculate likelihood of merged cluster       
    [logP_A_t,logP_Z_t]=evalLikelihood(Z_t,A,eta0p,eta0n,alpha);                

    % Zplit the merged cluster and calculate transition probabilties                
    % noc_tt=noc_t-1;
    setZ=setdiff(setZ,[ind1,ind2]);
    n_setZ=length(setZ);
    Z_tt=Z_t;
    Z_tt(clust1_t,:)=0;        
    comp=[clust1_t noc_t+1];               
    Z_tt(comp(1),ind1)=1;
    Z_tt(comp(2),ind2)=1;                

    % Reassign by restricted gibbs sampling
    if n_setZ>0
        for rep=1:2        
            [Z_tt,logP_A_tt,logP_Z_tt,logQ_trans,comp]=irm_gibbs(T,Z_tt,A,eta0p,eta0n,alpha,setZ(randperm(n_setZ)),comp);               
        end
        Force=[1 2]*Z([clust1 clust2],:);        
        [Z_tt,logP_A_tt,logP_Z_tt,logQ_trans]=irm_gibbs(T,Z_tt,A,eta0p,eta0n,alpha,setZ(randperm(n_setZ)),comp,Force);                        
    else
        logQ_trans=0;                   
    end
    % ADD TEMPERATURE HERE
%         a_merge=rand<exp(logP_A_t+logP_Z_t-logP_A-logP_Z+logQ_trans);
    a_merge=rand<exp((logP_A_t-logP_A + logP_Z_t-logP_Z)/T + logQ_trans);   

    if a_merge
      logP_A=logP_A_t;
      logP_Z=logP_Z_t;
      Z=Z_t;          
    end
end

function [Z,logP_A,logP_Z,logQ_trans,comp]=irm_gibbs(T,Z,A,eta0p,eta0n,alpha,JJ,comp,Force)        
if nargin<9
    Force=[];
end
if nargin<8
    comp=[];
end
logQ_trans=0;

clustFun=@betaln;

N=length(A);
[I,J]=size(A{1});
eN=ones(1,N);    
t=0;   
sumZ=sum(Z,2);
noc=length(sumZ);    

q=clustFun(eta0p,eta0n);
diag_const=q(1);
off_const=q(2);

Ap=eta0p(1)*eye(noc)+eta0p(2)*(ones(noc)-eye(noc));
An=eta0n(1)*eye(noc)+eta0n(2)*(ones(noc)-eye(noc));   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Count initial number of links and non-links between groups

n_link=zeros(noc,noc,N);
n_nonlink=n_link;


ZZT=sumZ*sumZ'-diag(sumZ);
for n=1:N        
    ZAZt=Z*A{n}*Z';
    n_link(:,:,n)=ZAZt+ZAZt';    
    n_link(:,:,n)=n_link(:,:,n)-0.5*diag(diag(n_link(:,:,n)));
    n_link(:,:,n)=n_link(:,:,n)+Ap;
    n_nonlink(:,:,n)=ZZT-ZAZt-ZAZt';
    n_nonlink(:,:,n)=n_nonlink(:,:,n)-0.5*diag(diag(n_nonlink(:,:,n)));
    n_nonlink(:,:,n)=n_nonlink(:,:,n)+An;   
    A{n}=logical(A{n}+A{n}');        
end
cluster_eval=clustFun(n_link,n_nonlink);
sum_cluster_eval=zeros(1,N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
for k=JJ          
    t=t+1;
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Remove effect of s_k
    ZA1k=zeros(noc,N);

    for n=1:N
        ZA1k(:,n)=Z*A{n}(:,k);
    end        
    sumZ=sumZ-Z(:,k); 
    nZA1k=sumZ*eN-ZA1k;

    d=find(Z(:,k));    % d = the cluster assigned to k    
    % Remove link counts generated from assigment Z(:,k)
    if ~isempty(d)
        n_link(:,d,:)=permute(n_link(:,d,:),[1 3 2])-ZA1k;        
        if N==1
            n_link(d,:)=n_link(d,:)-ZA1k';
        else
            n_link(d,:,:)=permute(n_link(d,:,:),[2 3 1])-ZA1k;
        end
        n_nonlink(:,d,:)=permute(n_nonlink(:,d,:),[1 3 2])-nZA1k;               
        if N==1
            n_nonlink(d,:)=n_nonlink(d,:)-nZA1k';                                      
        else
            n_nonlink(d,:,:)=permute(n_nonlink(d,:,:),[2 3 1])-nZA1k;                                      
        end
        n_link(d,d,:)=permute(n_link(d,d,:),[3 1 2])+ZA1k(d,:)';   
        n_nonlink(d,d,:)=permute(n_nonlink(d,d,:),[3 1 2])+nZA1k(d,:)';             
    end
    Z(:,k)=0;               

    if isempty(comp) % Distinguish between restricted and non-restricted sampling
        % Non-restricted sampling

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Remove singleton cluster            
        if sumZ(d)==0 
            v=1:noc;
            v(d)=[];
            length_d=length(d);
            d=[];
            noc=noc-length_d;                               
            P=sparse(1:noc,v,ones(1,noc),noc,noc+length_d);                        
            ZA1k=P*ZA1k;
            nZA1k=P*nZA1k; 
            Z=P*Z;                        
            sumZ=sumZ(v,1);    
            n_link=n_link(v,v,:);            
            n_nonlink=n_nonlink(v,v,:);            
            cluster_eval=cluster_eval(v,v,:);   
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate probability for existing communties as well as proposal cluster


        % Update cluster_eval without current node being assigned to any
        % clusters
        if ~isempty(d)
            cluster_eval(:,d,:)=clustFun(n_link(:,d,:),n_nonlink(:,d,:)); % removed the constant -betaln(Ap,An)))      
            cluster_eval(d,:,:)=permute(cluster_eval(:,d,:),[2 1 3]);                          
        end
        % Evaluate total likelihood for model without the node being
        % assigned
        for n=1:N
            sum_cluster_eval(n)=sum(sum(triu(cluster_eval(:,:,n))));  
        end

        e=ones(noc+1,1);       
        enoc=ones(1,noc);

        % Evaluate likelihood without contribution of d^th cluster
        if N==1
            sum_cluster_eval_d=e*sum_cluster_eval-[sum(cluster_eval,1)'; zeros(1,n)];
        else
            sum_cluster_eval_d=e*sum_cluster_eval-[permute(sum(cluster_eval,1),[2 3 1]); zeros(1,n)];
        end
        % Update likelihood when assigning node to each of the
        % clusters
        link=zeros(noc,noc+1,N);
        link(:,1:noc,:)=n_link+permute(ZA1k(:,:,enoc),[1 3 2]);
        link(:,noc+1,:)=ZA1k+eta0p(2);
        nolink=zeros(noc,noc+1,N);
        nolink(:,1:noc,:)=n_nonlink+permute(nZA1k(:,:,enoc),[1 3 2]);
        nolink(:,noc+1,:)=nZA1k+eta0n(2);                
        cluster_eval_d=clustFun(link,nolink);  
        if N==1
            sbeta=sum(cluster_eval_d,1)';
        else
            sbeta=permute(sum(cluster_eval_d,1),[2 3 1]);
        end
        sbeta(noc+1,:)=sbeta(noc+1,:)-noc*off_const;
        logQ=sum(sum_cluster_eval_d+sbeta,2); % removed the constant -betaln(Ap,An)))    



        % Zample from posterior     
        QQ=exp(logQ-max(logQ));
        % sumZ = number of nodes assigned to each cluster in Z
        weight=[sumZ; alpha]; 
        QQ=weight.*QQ; % proportionality of assignment to cluster Z, new cluster is given pseudocount alpha
        % temperature test, scale cluster attractiveness by
        % temperature, exaggerating the differences
        QQ=QQ.^(1/T);
        tmp = full(cumsum(QQ/sum(QQ))); % renormalize after application of temperature
        tmp(isnan(tmp)) = Inf;
        % Why this hack? Assignment to all other clusters is extremely
        % unlikely, so that alpha becomes infinite in comparison (with
        % Matlab precision). Then tmp becomes Inf/Inf=NaN. Obviously in
        % this context you'll want to assign to the new cluster, which
        % this hack solves.
        ind=find(rand<tmp,1,'first');  % ind = assignment for node k                           
        Z(ind,k)=1;   
        if ind>noc    %%% i.e. new cluster?                
            noc=noc+1;
            sumZ(noc,1)=0;
            n_link(:,noc,:)=eta0p(2);
            n_link(noc,:,:)=eta0p(2);            
            n_link(noc,noc,:)=eta0p(1);            
            n_nonlink(:,noc,:)=eta0n(2);                     
            n_nonlink(noc,:,:)=eta0n(2);                     
            n_nonlink(noc,noc,:)=eta0n(1);                                 
            cluster_eval(:,noc,:)=0;    
            cluster_eval(noc,:,:)=0;                
            cluster_eval_d1=permute(cluster_eval_d(:,noc,:),[1 3 2]);
            cluster_eval_d1(noc,:)=0;                                
            ZA1k(noc,:)=0;
            nZA1k(noc,:)=0;    
            logQf=logQ(noc)+N*(noc-1)*off_const;
        else
            cluster_eval_d1=permute(cluster_eval_d(1:noc,ind,:),[1 3 2]);                
            logQf=logQ(ind);
        end                        
    else            
        % Calculate probability for existing communties as well as proposal cluster
        if ~isempty(d)
            cluster_eval(:,d,:)=clustFun(n_link(:,d,:),n_nonlink(:,d,:)); % removed the constant -betaln(Ap,An)))                               
        end
        cluster_eval(d,:,:)=permute(cluster_eval(:,d,:),[2 1 3]);
        for n=1:N
            sum_cluster_eval(n)=sum(sum(triu(cluster_eval(:,:,n))));   
        end
        e=ones(2,1);

        if N==1
            sum_cluster_eval_d=e*sum_cluster_eval-sum(cluster_eval(:,comp))';                        
        else
            sum_cluster_eval_d=e*sum_cluster_eval-permute(sum(cluster_eval(:,comp,:)),[2 3 1]);                        
        end
        link=n_link(:,comp,:)+permute(ZA1k(:,:,e),[1 3 2]);                        
        nolink=n_nonlink(:,comp,:)+permute(nZA1k(:,:,e),[1 3 2]);            
        cluster_eval_d1=clustFun(link,nolink);
        if N==1
            sbeta=sum(cluster_eval_d1,1)';            
        else
            sbeta=permute(sum(cluster_eval_d1,1),[2 3 1]);            
        end
        logQ=sum(sum_cluster_eval_d+sbeta,2); % removed the constant -betaln(Ap,An)))  

        % Zample from posterior                        
        QQ=exp(logQ-max(logQ));
        weight=sumZ(comp);
        QQ=weight.*QQ;
        QQ=QQ/sum(QQ);
        % same test
        QQ=QQ.^(1/T);
        tmp = full(cumsum(QQ/sum(QQ))); % renormalize after application of temperature
        if isempty(Force)
            ind=find(rand<tmp,1,'first');
        else 
            ind=Force(k);
        end
        logQ_trans=logQ_trans+log(QQ(ind)+eps);
        Z(comp(ind),k)=1;  

        cluster_eval_d1=cluster_eval_d1(:,ind,:); 

        logQf=logQ(ind);
        ind=comp(ind);            
    end

    % Re-enter effect of new s_k        
    sumZ=sumZ+Z(:,k);

    n_link(:,ind,:)=permute(n_link(:,ind,:),[1 3 2])+ZA1k;
    if N==1
        n_link(ind,:)=n_link(ind,:)+ZA1k';
    else
        n_link(ind,:,:)=permute(n_link(ind,:,:),[2 3 1])+ZA1k;
    end
    n_link(ind,ind,:)=permute(n_link(ind,ind,:),[3 1 2])-ZA1k(ind,:)';
    n_nonlink(:,ind,:)=permute(n_nonlink(:,ind,:),[1 3 2])+nZA1k;        
    if N==1
        n_nonlink(ind,:)=n_nonlink(ind,:)+nZA1k';                
    else
        n_nonlink(ind,:,:)=permute(n_nonlink(ind,:,:),[2 3 1])+nZA1k;                
    end
    n_nonlink(ind,ind,:)=permute(n_nonlink(ind,ind,:),[3 1 2])-nZA1k(ind,:)';                
    cluster_eval(:,ind,:)=cluster_eval_d1;
    cluster_eval(ind,:,:)=cluster_eval_d1;

    % Remove empty clusters        
    if ~all(sumZ)
        d=find(sumZ==0);
        ind_d=find(d<comp);
        comp(ind_d)=comp(ind_d)-1;
        v=1:noc;
        v(d)=[];
        noc=noc-1;                               
        P=sparse(1:noc,v,ones(1,noc),noc,noc+1);                        
        Z=P*Z;                        
        sumZ=sumZ(v,1);   

        n_link=n_link(v,v,:);
        n_nonlink=n_nonlink(v,v,:);
        cluster_eval=cluster_eval(v,v,:); 
    end           

end              
noc=length(sumZ);

% add temperature here?
% posterior likelihood
logP_Z=noc*log(alpha)+sum(gammaln(full(sumZ)))-gammaln(J+alpha)+gammaln(alpha);    
logP_A=logQf-N*sum([noc noc*(noc-1)/2].*[diag_const off_const]); 
    
function [logP_A,logP_Z]=evalLikelihood(Z,A,eta0p,eta0n,alpha)

    N=length(A);
    
    [I,J]=size(A{1});
    noc=size(Z,1);
    sumZ=sum(Z,2);        
    logP_A=0;
    ii=find(triu(ones(noc)));      
  
    ZZT=sumZ*sumZ'-diag(sumZ);
    for n=1:N        
        ZAZt=Z*A{n}*Z';   
        n_link=ZAZt+ZAZt';    
        n_link=n_link-0.5*diag(diag(n_link));
        Ap=eta0p(1)*eye(noc)+eta0p(2)*(ones(noc)-eye(noc));
        An=eta0n(1)*eye(noc)+eta0n(2)*(ones(noc)-eye(noc));       
        n_link=n_link+Ap;
        n_nonlink=ZZT-ZAZt-ZAZt';
        n_nonlink=n_nonlink-0.5*diag(diag(n_nonlink));
        n_nonlink=n_nonlink+An;
        logP_A=logP_A+sum(betaln(n_link(ii),n_nonlink(ii)))-sum([noc noc*(noc-1)/2].*betaln(eta0p,eta0n));
    end
    logP_Z=noc*log(alpha)+sum(gammaln(full(sumZ)))-gammaln(J+alpha)+gammaln(alpha);  
    
function A = struct_conn_metropolis(T, A, N, Z, M, ap, an)

n = length(N);

Mpos = M{1};
Mneg = M{2};

linidx = find(triu(ones(n),1));
E = length(linidx);
for e=linidx(randperm(E))'
    Aprop = A;
    [i, j] = ind2sub([n n], e);
    
    Aprop(i,j) = 1-Aprop(i,j);
    Aprop(j,i) = 1-Aprop(j,i);
    
    dL = delta_log_dcm(N, ap, an, i, j, Aprop, A);    

    a = find(Z(:,i));
    b = find(Z(:,j));

    dZ = (1 - 2*A(i,j)) * log(Mpos(a,b) / Mneg(a,b));    
    alpha = dL + dZ;
    
    if rand <= min(1,exp(alpha)^(1/T))
        A = Aprop;    
    end     
end





