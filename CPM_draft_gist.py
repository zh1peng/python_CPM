# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:31:25 2019

@author: Zhipeng
"""


#get toy data
import glob
import shutil
import os
import pandas as pd
from scipy.io import loadmat
search_task = r'Z:\IMAGEN_MID_data\FU2_normal_1st_level\**\fc_map.mat'
tmp = list(glob.iglob(search_task, recursive=True))
subid = [os.path.basename(os.path.dirname(filen)) for filen in tmp]
fc_file=pd.DataFrame([subid,tmp],['ID','fc_file']).T
fc_file['ID']=fc_file['ID'].astype(int)
fc_file=fc_file.set_index('ID')
SS=FU2_data[['ss_sum']].dropna()
SS_fc=fc_file.join(SS).dropna()
toy_data=SS_fc.drop(SS_fc.index[-1161:])


toy_data=SS_fc

y1=[]
y2=[]
X=[]
for idx, subi in enumerate(toy_data.index.tolist()):
    print('extracting {} subject fc map...'.format(idx))
    mat2load=toy_data.loc[subi,'fc_file']
    tmp=loadmat(mat2load)
    conn=tmp['con_PPI_final']
#    inds = np.where(np.isnan(conn))
#    conn[inds]=0
    upper_idx=np.triu_indices(conn.shape[0],1)
    upper_fc=conn[upper_idx]
    y1.append(toy_data.loc[subi,'ss_sum'])
    X.append(upper_fc)
X1=np.vstack(X)
y1=np.vstack(y1)

num_cols = X1.shape[1]
rng = np.arange(num_cols)
new_cols = ['con_' + str(i) for i in rng]
#df.columns = new_cols[:num_cols]


fs_df=pd.DataFrame(X1,columns=new_cols,index=SS_fc.index)
y_df=pd.DataFrame(y1, columns=['y_ss'],index=SS_fc.index)
tmp=SS_fc_cov.drop(columns=['fc_file','ss_sum'])
tmp.columns=['flag_'+str(col) for col in tmp.columns]
# ensure the length of the new columns list is equal to the length of df's columns
fs_cov_df=fs_df.join(tmp).dropna()
fs_cov_df=fs_cov_df.join(y_df).dropna()
fs_cov_df.to_csv('Y:\zhipeng EEG preprocessing\ML_python\ML_con_166_1251_only_SS.csv')



import scipy
def fc_behav_corr(fc,y):
    if np.ndim(y)==1:
        y_pred=y_pred[:,None]
    fc_dim=fc.shape[-1]
    r_value=np.empty((fc.shape[-1]))*np.nan
    p_value=np.empty((fc.shape[-1]))*np.nan
    for fc_i in np.arange(fc_dim):
        r_value[fc_i], p_value[fc_i]=scipy.stats.pearsonr(y,fc[:,fc_i][:,None])
    return r_value, p_value

r,p=corr_y(X1, y1)
p_mask=p<0.5
p_mask.astype(int)
p_masks=np.repeat(p_mask[None,:],100,axis=0)
sig_edge=fc[p_masks]


def apply_mask(row_fc, mask):
    row_fc=row_fc[:,None]
    masked_fc=row_fc[mask]
    return masked_fc

def sum_edge(row_fc,method='pos'):
    row_fc=row_fc[:,None]
    if method=='pos':
        result=row_fc[row_fc>0].sum()
    elif method=='neg':
        result=row_fc[row_fc<0].sum()
    return result

sig_edge=np.apply_along_axis(apply_mask,1,fc,mask=p_mask)
pos_sum=np.apply_along_axis(sum_edge,1,sig_edge,'pos')
neg_sum=np.apply_along_axis(sum_edge,1,sig_edge,'neg')

all_sum=np.vstack([pos_sum,neg_sum]).T

#CPM https://www.nature.com/articles/nprot.2016.178


#1. positive and negative transformer
#2.  

from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.linear_model import LinearRegression

def _apply_mask(row_fc, mask):
    row_fc=row_fc[:,None]
    masked_fc=row_fc[mask]
    return masked_fc
def _sum_edge(row_fc,method='pos'):
    row_fc=row_fc[:,None]
    if method=='pos':
        result=row_fc[row_fc>0].sum()
    elif method=='neg':
        result=row_fc[row_fc<0].sum()
    return result
def _fc_behav_corr(fc,y):
    if np.ndim(y)==1:
        y=y[:,None]
    fc_dim=fc.shape[-1]
    r_value=np.empty((fc.shape[-1]))*np.nan
    p_value=np.empty((fc.shape[-1]))*np.nan
    for fc_i in np.arange(fc_dim):
        r_value[fc_i], p_value[fc_i]=scipy.stats.pearsonr(y,fc[:,fc_i][:,None])
    return r_value, p_value

class select_edge(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, threshold=0.5, method='pearson'):
        self.p_mask=[]
        self.sig_edge=[]
        self.threshold=threshold
        self.method=method     
    def fit(self, X, y):
        _,sig_p=_fc_behav_corr(X, y)
        p_mask=sig_p<self.threshold
        self.p_mask=p_mask[:,None]
    def transform(self, X):
        sig_edge=np.apply_along_axis(_apply_mask,1,X,mask=self.p_mask)
        pos_sum=np.apply_along_axis(_sum_edge,1,sig_edge,'pos')
        neg_sum=np.apply_along_axis(_sum_edge,1,sig_edge,'neg')
        all_sum=np.vstack([pos_sum,neg_sum]).T
        self.sig_edge=sig_edge
        return all_sum
    def get_p_mask(self):
        return self.p_mask
    def get_sig_edge(self):
        return self.sig_edge

#tf=select_edge()
#tf.fit(X1,y1)
#tf.transform(X1)

class CPM(BaseEstimator):
    def __init__(self,threshold=0.5):
        self.threshold=threshold
        self.edge_tf=select_edge(threshold=self.threshold)
        self.clf=LinearRegression()
    def fit(self, X, y):
        self.edge_tf.fit(X,y)
        fs2use=self.edge_tf.transform(X)
        self.clf.fit(fs2use,y)
        return self
    def predict(self, X):
        fs2use=self.edge_tf.transform(X)
        pred=self.clf.predict(fs2use)
        return pred
    
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict
clf=CPM(threshold=0.001) 
y_pred = cross_val_predict(clf, X1, y1,cv=5,n_jobs=-1)
    
import scipy        
scipy.stats.pearsonr(y1,y_pred)     
(array([-0.00176464]), array([0.95008406])) 0.5
(array([-0.00960228]), array([0.73336567])) 0.01


# Test with activiation
activation_data=pd.read_csv(r'Y:\zhipeng EEG preprocessing\ML_python\ML_activiation_166sphere_1211_tlfb_ss.csv')       
fs=np.array(activation_data[[col for col in activation_data.columns if col.startswith('ROI')]])
y=activation_data['y_binge_days']
clf=CPM(threshold=0.1)     
y_pred = cross_val_predict(clf,fs, y,cv=10,n_jobs=-1)    
scipy.stats.pearsonr(y,y_pred)     



fs=np.array(final[[col for col in final.columns if col.startswith('ROI')]])
y=np.array(final['y_ss_sum'])
  
