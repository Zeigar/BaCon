function dLogLikelihood = delta_log_dcm(N, ap, an, k, l, gProposal, gCurrent)
% Calculates the change in log-likelihood after an edge flip in the
% MCMC-Metropolis sampler. Derivation of the equation is shown in (B.3) in
% [1]. 
%
% Required parameters are the nxn matrix of streamline counts N, the
% hyperparameters of the Dirichlet multinomial compound distribution ap and
% an, the (non)-edge (k,l), the proposed graph gProposal and the current
% graph gCurrent.
%
% References:
%
% [1] Hinne, M., Heskes, T., Beckmann, C.F., & van Gerven, M.AJ. (2013).
% Bayesian inference of structural brain networks. NeuroImage, 66C,
% 543–552.
%
% Last modified: April 8th, 2014

nodecount = length(N);

e_kl = gProposal(k, l);
e_lk = e_kl;

anew_kl = e_kl * ap + (1 - e_kl) * an;
aold_kl = (1 - e_kl) * ap + e_kl * an;

anew_sum_k = sum(gProposal(k,:)) * ap + (nodecount - sum(gProposal(k,:))) * an; 
aold_sum_k = sum(gCurrent(k,:)) * ap + (nodecount - sum(gCurrent(k,:))) * an; 

term1 = gammaln(anew_kl + N(k, l)) - gammaln(aold_kl + N(k, l));
term2 = gammaln(anew_kl) - gammaln(aold_kl);
term3 = gammaln(anew_sum_k) - gammaln(aold_sum_k);
term4 = gammaln(anew_sum_k + sum(N(k, :))) - gammaln(aold_sum_k + sum(N(k, :)));

anew_lk = e_lk * ap + (1 - e_lk) * an;
aold_lk = (1 - e_lk) * ap + e_lk * an;

anew_sum_l = sum(gProposal(l,:)) * ap + (nodecount - sum(gProposal(l,:))) * an;
aold_sum_l = sum(gCurrent(l,:)) * ap + (nodecount - sum(gCurrent(l,:))) * an;

term5 = gammaln(anew_lk + N(l, k)) - gammaln(aold_lk + N(l, k));
term6 = gammaln(anew_lk) - gammaln(aold_lk);
term7 = gammaln(anew_sum_l) - gammaln(aold_sum_l);
term8 = gammaln(anew_sum_l + sum(N(l, :))) - gammaln(aold_sum_l + sum(N(l, :)));

dLogLikelihood = term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8;
