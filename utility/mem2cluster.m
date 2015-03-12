function Z = mem2cluster(mem)

n=length(mem);

Z = zeros(max(mem),n);
linind = sub2ind(size(Z),mem',1:n);
Z(linind)=1;