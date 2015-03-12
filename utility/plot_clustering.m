function [order, val, map, A] = plot_clustering(A,Z)

nNodes = length(A);

nodelabels = zeros(nNodes,2);
for k=1:nNodes
  ix = find(Z(:,k)); 
  nodelabels(k,:) = [ix k]';
end

[val, idx]=sort(nodelabels(:,1));

nodelabelssorted=nodelabels(idx,:);
order=nodelabelssorted(:,2);


[V,W] = find(A);
for k=1:length(V)
   v = V(k);
   w = W(k);
   zv = nodelabels(v,1);
   zw = nodelabels(w,1);
   if zv == zw
       A(v,w)=zv+1;
       A(w,v)=zw+1;
   end
end

basemap=colormap(jet(size(Z,1)));
nColors = size(basemap,1);
K = size(Z,1);

map = zeros(K+2,3);

map(1,:) = [1 1 1];
map(2,:) = [0 0 0];

if K > nColors
    basemap = repmat(basemap,ceil(K/nColors),1);
end

map(3:K+2,:) = basemap(1:K,:);
A = A(order,order);
A = A+diag(val+1);

for k=1:K
    nk = find(val==k);
    korder = symrcm(A(nk,nk));
    A(nk,nk) = A(nk(korder),nk(korder));
    order(nk) = order(nk(korder));
end

imagesc(A); colormap(map); axis square;

end

function map = gorgeous()
    map = [0 104 132;
    0 144 158;
    137 219 236;
    237 0 38;
    250 157 0;
    255 208 141;
    176 0 81;
    246 131 112;
    254 171 185;
    110 0 108;
    145 39 143;
    207 151 215;
    91 91 91;
    212 212 212;]/255;
end

function map = basic()
    map = [178 31 53;
    216 39 53;
    255 116 53;
    255 161 53;
    255 203 53;
    255 240 53;
    0 117 58;
    0 158 71;
    22 221 53;
    0 82 165;
    0 121 231;
    0 169 252;
    104 30 126;
    125 60 181;
    189 122 246;] / 255;
end