function Wexp = weighted_exp(samples, weights)
[p,~,T] = size(samples);
Wexp = zeros(p);
max_w = max(weights);
for i=1:T-1
    w = exp(weights(i) - max_w);
    Wexp = Wexp + samples(:,:,i) * w;
end
Wexp = Wexp / sum(exp(weights - max_w));