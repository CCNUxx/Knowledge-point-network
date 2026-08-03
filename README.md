# Introduction

This project contains some core source code and example data for the paper titled "How to quantify an examination? Evidence from physics examinations via complex networks".

The core codes run perfectly on Python 3.10, and the libraries used are as follows:

matplotlib==3.9.1

networkx==3.3

numpy==1.26.4

openpyxl==3.1.4

pandas==2.2.2

We do not guarantee normal operation on other versions.

Note: the original examination data are currently being used in the authors' ongoing doctoral research and therefore cannot be fully shared at this stage. Only a subset of core example data is provided here. Researchers with reasonable requests may contact the corresponding author to obtain the original data.

# How to use

All example data is stored in the "./data" folder. You just need to run the "main.py" file.

The "./data" folder contains:

* "13V2.xlsx" and "17V1.xlsx": two example exam papers (question, knowledge point) used to demonstrate how a KPN is constructed.
* "First stage\_matrix.xlsx": the IKPN of the first stage (2006-2010).

"main.py" performs the following steps:

1. Build the KPN from the example exam paper.
2. Compute the topological metrics for both the example KPN and the first stage network.
3. Rank the top-15 knowledge points by eigenvector centrality.
4. Run a perturbation-based robustness analysis on the first stage network.

If you use any of the related data or codes, please cite this paper: Xia, M., Su, Z., Deng, WB. et al. How to quantify an examination? Evidence from physics examinations via complex networks. Humanit Soc Sci Commun (2026). https://doi.org/10.1057/s41599-026-07735-6.

