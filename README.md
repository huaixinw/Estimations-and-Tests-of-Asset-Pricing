# Basic Methods for Estimating and Tesing Factor Pricing Model
Fama-MacBeth regression, Newey-West adjustment, and GMM adjustment (Shanken, 1992)

This project aims to learn the basic tools of estimating and testing asset pricing models by replicating main results of 
He, Z., Kelly, B., & Manela, A. (2017). Intermediary asset pricing: New evidence from many asset classes. Journal of Financial Economics, 126(1), 1-35.

The data set is obtained from the homepage of Prof.Asaf Manela (http://apps.olin.wustl.edu/faculty/manela/data.html) .File "He_Kelly_Manela_Factors_And_Test_Assets.csv" is used here. In addition, the Julia code of Prof.Asaf Manela is of great help in our replication.

The replication target is Table 5 of the above paper, which contains estimation of betas and standard errors. The main methods are Fama-MacBeth regression and Shanken-GMM adjustment (1992), I also add a simple function for Newey-West adjustment.

NOTE: 
1. For now, the code only contains the "estimation" part, the "testing" part is in processing.
2. before running the code, one should download the data from the above link (Prof.Asaf Manela's page).

