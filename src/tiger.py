import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import pandas as pd

def pbso_integrated(X_back,W_back,Npsign,NpCT):
    """Overall function for optimizing function.
    
    X_back: matrix with data for covariates that might affect tiger presence
    W_back: matrix with data for covariates that might bias presence-only data
    Npsign: single value sign survey
    NpCT: single value camera trap
       
    Returns dataframe of coefficients dataframe (parameter name, value, standard error),
    convergence, message for optimization, and value of negative log-likelihood"""
       
    beta_names=list(X_back)
    beta_names[0]='beta0'
    alpha_names=list(W_back)
    alpha_names[0]='alpha0'
    psign_names=['p_sign_{0}'.format(i) for i in range(0,Npsign)]
    pcam_names=['p_cam_{0}'.format(i) for i in range(0,NpCT)]
    par_names=beta_names+alpha_names+psign_names+pcam_names
    paramGuess=np.zeros(len(par_names))
    fit_pbso = minimize(negLL_int,paramGuess,method='BFGS',options={'gtol':1e-08})
    se_pbso = np.zeros(len(fit_pbso.x))
    #if fit_pbso.success==True:
    #    se_pbso = np.sqrt(np.diag(fit_pbso.hess_inv))
    tmp = {'Parameter name':par_names,'Value':fit_pbso.x,'Standard error':se_pbso[0]}
    tmp_df=pd.DataFrame(tmp,columns=['Parameter name','Value','Standard error'])
    p = {'coefs':tmp_df, 'convergence':fit_pbso.success,'optim_message':fit_pbso.message,'value':fit_pbso.fun}
    return p

def negLL_int(par):
     """Calculates the negative log-likelihood of the function.
     
     Par: array list of parameters to optimize
     
     Returns single value of negative log-likelihood of function"""
     
     beta = par[0:Nx]
     alpha = par[Nx:Nx+Nw]
     p_sign = expit(par[Nx+Nw:Nx+Nw+Npsign])
     p_cam = expit(par[Nx+Nw+Npsign:Nx+Nw+Npsign+NpCT])
     lambda0 = np.exp(np.dot(np.array(X_back),beta))
     psi = 1. - np.exp(-lambda0)
     tw = np.dot(np.array(W_back),alpha)
     p_thin = expit(tw)
     zeta = np.empty((len(psi),2))
     zeta[:,0] = 1.- (psi)
     zeta[:,1] = np.log(psi)
     
     for i in range(0,len(CT['det'])):
         zeta[CT['cell'][i]-1,1] = zeta[CT['cell'][i]-1,1]+(CT['det'][i])*np.log(p_cam[CT['PI'][i]-1])+(CT['days'][i]-CT['det'][i])*np.log(1.-p_cam[CT['PI'][i]-1])
     
     for j in range(0,len(sign['dets'])):
         zeta[sign['cell'][j]-1,1] = zeta[sign['cell'][j]-1,1]+(sign['dets'][j])*np.log(p_sign[sign['survey.id'][j]-1])+(sign['reps'][j]-sign['dets'][j])*np.log(1.-p_sign[sign['survey.id'][j]-1])

     one=sign[sign['dets']>0]['cell']
     two=CT[CT['det']>0]['cell']
     known_occ=list(set(one.append(two)))

     zeta[np.array(known_occ)-1,0]=0
     
     lik_so = []
     
     for i in range(0,len(zeta[:,0])):
         if zeta[i,0]==0:
             lik_so.append(zeta[i,1])
         else:
             lik_so.append(np.log(zeta[i,0])+zeta[i,1])
     
     nll_po = -1.*(-1.*sum(lambda0*p_thin)+sum(np.log(lambda0[po_data-1]*p_thin[po_data-1])))
     nll_so = -1.*sum(lik_so)

     return nll_po[0]+nll_so
 
def predict_surface(par):
     """Create predicted probability surface for each grid cell.
     
     Par: list of parameter values that have been optimized to convert to probability surface
     
     Returns data frame that includes grid code, grid cell number and predicted probability surface for each grid cell"""

     par=np.array(par)
     beta = par[0:Nx]
     p_sign = expit(par[Nx+Nw:Nx+Nw+Npsign])
     p_cam = expit(par[Nx+Nw+Npsign:Nx+Nw+Npsign+NpCT])
     lambda0 = np.exp(np.dot(np.array(X_back),beta))
     psi = 1. - np.exp(-lambda0)
     zeta = np.empty((len(psi),2))
     zeta[:,0] = 1.- (psi)
     zeta[:,1] = np.log(psi)
     
     for i in range(0,len(CT['det'])):
         zeta[CT['cell'][i]-1,1] = zeta[CT['cell'][i]-1,1]+(CT['det'][i])*np.log(p_cam[CT['PI'][i]-1])+(CT['days'][i]-CT['det'][i])*np.log(1.-p_cam[CT['PI'][i]-1])
     
     for j in range(0,len(sign['dets'])):
         zeta[sign['cell'][j]-1,1] = zeta[sign['cell'][j]-1,1]+(sign['dets'][j])*np.log(p_sign[sign['survey.id'][j]-1])+(sign['reps'][j]-sign['dets'][j])*np.log(1.-p_sign[sign['survey.id'][j]-1])

     one=sign[sign['dets']>0]['cell']
     two=CT[CT['det']>0]['cell']
     known_occ=list(set(one.append(two)))

     zeta[np.array(known_occ)-1,0]=0
     cond_psi=[zeta[i,1]/sum(zeta[i,:]) for i in range(0,len(psi))]
     
     cond_prob=1.-(1.-np.exp(np.multiply(-1,cond_psi)))
     grids=[i for i in range(1,len(psi)+1)]
     temp={'gridcode':griddata['gridcode'],'grid':grids,'condprob':cond_prob}
     prob_out=pd.DataFrame(temp,columns=['gridcode','grid','condprob'])
     
     return prob_out
    
#inputs
griddata = pd.read_csv(r'C:\Users\Jamie\Documents\tiger\python_convert\model.inputs\griddata.csv') 
CT = pd.read_csv(r'C:\Users\Jamie\Documents\tiger\python_convert\model.inputs\CT.csv')
po_data = pd.read_csv(r'C:\Users\Jamie\Documents\tiger\python_convert\model.inputs\po_data.csv')
sign = pd.read_csv(r'C:\Users\Jamie\Documents\tiger\python_convert\model.inputs\sign.csv')  
Nx=3
Nw=3
Npsign=1
NpCT=1
W_back=griddata[['tri','distance_to_roads']]
W_back.insert(0,'Int',1)
X_back=griddata[['woody_cover','hii']]
X_back.insert(0,'Int',1)

m = pbso_integrated(X_back,W_back,Npsign,NpCT)
probs = predict_surface(m['coefs']['Value'])
    
    