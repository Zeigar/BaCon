BaConToolbox
=====

Bayesian Connectomics toolbox

Estimate posterior distributions for
- Structural connectivity, using probabilistic tractography count data.
	- Prior on network density or
	- rior on clustering.
- Functional connectivity using a structural estimate as constraint, using fMRI BOLD signal time series.
- Functional connectivity using a dynamic constraint (i.e. learn conditional independence and partial correlation simultaneously), using fMRI BOLD signal time series.
- Functional and structural connectivity simultaneously, using fMRI BOLD signal time series and probabilistic tractography count data.

See demo.m for working examples of all provided scripts.

Related literature

[1] Max Hinne, Tom Heskes, Christian Beckmann and Marcel van Gerven, 2013. Bayesian inference of structural brain networks. NeuroImage 66, pp. 543-552.

[2] Ronald Janssen, Max Hinne, Tom Heskes and Marcel van Gerven, 2014. Quantifying Uncertainty in Brain Network Measures using Bayesian Connectomics. Frontiers in Computational Neuroscience 8 (126).

[3] Max Hinne, Luca Ambrogioni, Ronald Janssen, Tom Heskes and Marcel van Gerven, 2014. Structurally-informed Bayesian functional connectivity analysis. NeuroImage 68, pp. 294-305.

[4] Max Hinne, Alex Lenkoski, Tom Heskes and Marcel van Gerven, 2014. Efficient sampling of Gaussian graphical models using conditional Bayes factors. Stat 3, pp. 326-336.

[5] Max Hinne, Matthias Ekman, Ronald Janssen, Tom Heskes and Marcel van Gerven, 2015. Probabilistic clustering of the human connectome identifies communities and hubs. PLoS ONE 10(1), pp. e0117179.
