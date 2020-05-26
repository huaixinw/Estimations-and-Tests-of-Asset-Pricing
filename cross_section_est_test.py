# -*- coding: utf-8 -*-
"""
Fama-Macbeth regression and GMM adjustment
replication of:
    He, Z., Kelly, B., & Manela, A. (2017). 
    Intermediary asset pricing_New evidence from many asset classes. JFE.
@author: huaixin
@e-mail: wanghx.19@pbcsf.tsinghua.edu.cn
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

#------------------------------------------------------------------------------
# Load and specify data  
testdata = pd.read_csv('He_Kelly_Manela_Factors_And_Test_Assets.csv')

assets_equity = pd.DataFrame()
assets_USbond = pd.DataFrame()
assets_SovBond = pd.DataFrame()
assets_option = pd.DataFrame()
assets_cds = pd.DataFrame()
assets_com = pd.DataFrame()

for var in set(testdata.columns):
    if 'FF25' in var:
        assets_equity[var] = testdata[var]
    elif 'US_bonds' in var:
        assets_USbond[var] = testdata[var]
    elif 'Sov_bonds' in var:
        assets_SovBond[var] = testdata[var]
    elif 'Options' in var:
        assets_option[var] = testdata[var]
    elif 'CDS' in var:
        assets_cds[var] = testdata[var]
    elif 'Commod' in var:
        assets_com[var] = testdata[var]
    else:
        continue

# factors: the pricing factors of this paper
factors = testdata[['mkt_rf','intermediary_capital_risk_factor']]

# specify risk-free rate
riskfree = testdata['rf']

#------------------------------------------------------------------------------
# Define the function
# The function that contains the basic methods of tests of cross-section returns
def cross_sectional_test(assets, factors, riskfree,
                         nwlag=['auto','b'],
                         cross_intercept=True):
    ###### DESCRIPTION OF THE FUNCTION ######
    # This function provides classical asset pricing model estimation and tests 
    # including beta estimation, Fama-MacBeth regression, newey-west adjustment
    # and GMM standard error
    ###### PARAMETERS ######
    # dates:    
    #   Dataframe format. data dates
    # assets:  
    #   Dataframe format. Testing assets, FamaFrench 25 portfolios, 
    #   anomaly-based portfolios, for instance.
    # factors:  
    #   Dataframe format. Factor data of your asset pricing model
    # riskfree: 
    #   Dataframe format. Riskfree rate.
    #-------------
    # NOTE: Dates is not necessary. The rows of assets, factors and riskfree must
    #       correspond; if using lagged risk-free rate, the 'riskfree' must be the 
    #       lagged data and corresponds each rows of assets and factors.
    #-------------
    # nwlag: 
    #   List, [integer] or ['auto','kernel'] The lag number of newey-west adjustment.
    #   When using given nember of lags, use [integer], nwlag=[6] for instance;
    #   when using nember of lags calculated from the length of periods, use
    #   ['auto', kernel], where kernel='b' denotes Bartlett kernel and kernel='q' 
    #   denotes quadratic spectral kernel.
    # cross_intercept:
    #   Bool. Whether adding intercept term in the cross-sectional regression of Fama-MacBeth
    #   method, default Ture.
    
    
    # drop missing values and reset index for regression
    assets = assets.dropna(axis=0)
    factors = factors.iloc[assets.index].reset_index(drop=True)
    riskfree = riskfree.iloc[assets.index].reset_index(drop=True)
    assets = assets.reset_index(drop=True)
    
    # calculate excess returns of testing assets
    # get the number of testing assets and time period as N and T
    # number of explanation variables(number of factors, exclude the intercept)
    excess = assets.sub(np.array(riskfree),axis='rows') 
    T, N = assets.shape    
    k = factors.shape[1]
    
    ############################################################# 
    # Fama-MecBath Regression 
    ############################################################# 
    
    #### First step: time-series regression to estimate beta 
    # the first step regression contains the intercept in regression
    # this intercept is also used for GRS test
    
    # for each testing asset, run time-series regression
    reg1 = sm.OLS(endog=excess, exog=sm.add_constant(factors)).fit()
    param1 = reg1.params.T
    a = param1['const']
    beta = param1.drop(['const'],axis=1)
    
    # Newey-West Adjustment
    # Econometric analysis (8th edition) by Greene, page 999.
    # Empirical asset pricing by Bail et al., page 7.
    if nwlag[0] == 'auto':
        if nwlag[1] == 'b':
            # Bartlett kernel
            nw_lags = int(4*(T/100)**(2/9))+1
        else:
            # quadratic spectral kernel
            nw_lags = int(4*(T/100)**(4/25))+1
    else:
        nw_lags = nwlag[0]
                
    error = reg1.resid
    std_nw = pd.DataFrame()
    X = sm.add_constant(factors).to_numpy()
    for code in excess.columns:
        e2 = np.square(error[code].to_numpy())
        S0 = X.T @ np.diag(e2) @ X
        SLT = np.zeros((k+1,k+1))
        for l in range(0,nw_lags):
            error_lu = error[code][l:]
            error_ld = error[code][0:T-l]
            wl = 1-l/(1+nw_lags)
            etel = wl*error_lu.mul(error_lu,axis='rows').to_numpy()
            X_lu = X[l:]
            X_ld = X[0:T-l]
            S1 = X_lu.T @ np.diag(etel) @ X_ld          
            S2 = X_ld.T @ np.diag(etel) @ X_lu
            SLT = SLT+S1+S2
        
        S_nw = 1/T*(S0+SLT)
        std_nw = std_nw.append(pd.DataFrame(
                                            np.matrix(
                                                np.sqrt(np.diag(S_nw))
                                                    )
                                            )
                                )
    std_nw.columns = param1.columns
    std_nw = std_nw.reset_index(drop=True)
    tvalues_nw = param1.div(std_nw)    
               
    ### second step: cross_sectional regression for each period and average 
    # whether the second step regression contains the intercept depends on the specific and theory
    exp_ret = reg1.fittedvalues
    
    # for each period, run cross-sectional regression
    param2 = pd.DataFrame()
    # parameter cross_intercept decides whether add intercept in the cross-sectional regression
    if cross_intercept==True:
        for t in assets.index:
            eret = exp_ret[t:t+1].T.reset_index(drop=True)
            reg2 = sm.OLS(endog=eret, exog=sm.add_constant(beta)).fit()
            param2 = param2.append(pd.DataFrame(reg2.params).T)
    else:
        for t in assets.index:
            eret = exp_ret[t:t+1].T.reset_index(drop=True)
            reg2 = sm.OLS(endog=eret, exog=beta).fit()
            param2 = param2.append(pd.DataFrame(reg2.params).T)
        
    gamma = param2.mean()
    
    ############################################################# 
    # GMM standard errors (Shanken 1992)
    ############################################################# 
    # Cochrane p241, Shanken(1992)
    if cross_intercept==True:
        K = k+1
        g = gamma[1:].to_numpy()
        b = np.block([
            [np.ones((N,1)), beta.to_numpy()]
            ])
    else:
        K = k
        g = gamma.to_numpy()
        b = beta.to_numpy()

    f = factors.to_numpy()
    fit = reg1.fittedvalues
    fit.columns = excess.columns
    e = (excess-fit).to_numpy() # epsilon from eq.(12.9), p235
    
    # the a matrix, see the general GMM formulas, p202
    # this matrix chooses which moment conditions are set to zero in estimation
    am = np.block([
        [np.eye((k+1)*N),         np.zeros(((k+1)*N,N))],
        [np.zeros((K,(k+1)*N)), b.T                  ]
        ])
    
    # the d matrix
    # sensitivity of the moment conditions to the parameters
    Ef = factors.mean().to_numpy()
    Eff = np.dot(f.T,f)/T
    
    d = np.block([
        [1,               np.matrix(Ef)],
        [np.matrix(Ef).T, Eff          ],
        ])
    d = np.kron(d,np.eye(N))
    d = -np.block([
        [d, np.zeros(((k+1)*N,K))],
        [np.zeros((N,N)), np.kron(np.matrix(g),np.eye(N)), b]
        ])
    
    # S matrix: long-run covariance matrix of the moments
    u = e
    for i in range(0,k):
        u = np.block([
            [u, pd.DataFrame(e).mul(f[:,i],axis='rows').to_numpy()]
            ])
           
    preret = np.matrix(np.dot(b,param2.mean().to_numpy()))
    
    u = np.block([
        [u, excess.to_numpy() - preret]
        ])
    
    u_demean = u - u.mean(axis=0)
    S = np.dot(u_demean.T,u_demean)/T
    
    # formula of the covariance matrix of parameters, p202
    # the last k by k block matrix is for risk premium(gamma)
    # if containing intercept, the K by K block matrix
    adinv = np.linalg.inv(np.dot(am,d))
    wm = np.dot(am,S)
    wm = np.dot(wm,am.T)
    cov_gmm = 1/T*(adinv @ am @ S @ am.T @ adinv.T)
    std_g_gmm = np.sqrt(np.diag(cov_gmm[-1-(K-1):,-1-(K-1):]))
    t_g_gmm = gamma/std_g_gmm
    
    gamma_percentage = gamma*100
    
    est_result = pd.DataFrame([gamma_percentage,t_g_gmm],index=['coefficient (%)','t-value'])
    
    return est_result

#------------------------------------------------------------------------------   
# Example   
# use equity data as an example
assets = assets_equity
result = cross_sectional_test(assets=assets_equity, factors=factors, riskfree=riskfree,
                              nwlag=['auto','b'],
                              cross_intercept=True)
print(result)
