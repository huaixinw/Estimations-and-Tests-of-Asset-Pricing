# Asset-Pricing-Tests
Fama-MacBeth regression, Newey-West adjustment, and GMM adjustment (Shanken, 1992)

This project aims to learn the basic tools of asset pricing tests by replicating main results of 
He, Z., Kelly, B., & Manela, A. (2017). Intermediary asset pricing: New evidence from many asset classes. Journal of Financial Economics, 126(1), 1-35.

The data is obtained from the homepage of Prof.Asaf Manela (http://apps.olin.wustl.edu/faculty/manela/data.html) .File "He_Kelly_Manela_Factors_And_Test_Assets.csv" is used here. In addition, we replicate using Python 3.7, the Julia code of Prof.Asaf Manela is of great help in our replication.

The replication target is Table 5 of the above paper, which contains estimation of betas and standard errors. The main methods are Fama-MacBeth regression and Shanken-GMM adjustment (1992), we also add a simple function for Newey-West adjustment.

NOTE: 
1. file "asset pricing tests add if intercept.py" is the replication code.
2. before running the code, one should download the data from the above link.
