import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.linear_model import LinearRegression

def _apply_mask(row_fc, mask):
'''
apply 'sig' mask to the fc
mask: binary mask
row_fc: flatten fc matrix
'''
    row_fc=row_fc[:,None]
    masked_fc=row_fc[mask]
    return masked_fc

def _sum_edge(row_fc,method='pos'):
'''
sum pos/neg edges
'''
    row_fc=row_fc[:,None]
    if method=='pos':
        result=row_fc[row_fc>0].sum()
    elif method=='neg':
        result=row_fc[row_fc<0].sum()
    return result

def _fc_behav_corr(fc,y):
'''
correlate behavioural data with the predictors (i.e., edges)
'''
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
    Feature selection:
    Selection 'sig' edges that have p_corr<threshold(=0.5) 
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
'''
implement the edge selection in CV loops
'''
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
 
#clf=CPM(threshold=0.1)     
#y_pred = cross_val_predict(clf,X, y,cv=10,n_jobs=-1)    
#scipy.stats.pearsonr(y,y_pred)    
