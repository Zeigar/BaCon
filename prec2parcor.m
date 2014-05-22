function R = prec2parcor(K,rescale)
% Scale precision matrix to partial correlation.
%
% Last modified: April 8th, 2014

n = length(K);
R = zeros(n);

for i=1:n
   for j=i+1:n
       R(i,j) = -K(i,j) / sqrt(K(i,i)*K(j,j));
   end
end

if nargin > 1 && rescale
    R = R+R';
else
    R = R+R'+eye(n);
end

