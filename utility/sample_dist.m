function [pr, ix] = sample_dist(samples)

[p,~,T] = size(samples);

Gs = zeros(p,p,1);
Gs(:,:,1) = samples(:,:,1);
pr = 1;
ix = 1;

for t=2:T
    G = samples(:,:,t);
    existing = find(all(all(bsxfun(@eq, Gs, G))));
    if isempty(existing)
        pr(end+1) = 1;
        Gs = cat(3,Gs,G);
        ix = [ix; t];
    else
        pr(existing) = pr(existing) + 1;
    end
end

pr = pr / T;
