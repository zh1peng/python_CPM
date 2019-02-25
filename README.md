# python_CPM
## python version of the CPM
This is the python version of the CPM described [Nature Protocol](https://www.nature.com/articles/nprot.2016.178). 
I just followed the idea doing feature selection with sig correlation between edges (this also could be other brain data I think) and behavioural measures. The magic thing is that the sum of the sig edges would work in predicting Y. But I did not find any results using gPPI-based FC during reward anticipation in MID task to predict sensation seeking measured by SUPRS or alcohol use measured by timeline follow-back (TLFB).

I did not read any matlab codes shared by the author, so this is very rough implementation of CPM by my understanding.


## [CPM_draft_gist](https://github.com/zh1peng/python_CPM/blob/master/CPM_draft_gist.py)
Draft codes that I used to make the clf

## [python_CPM.py](https://github.com/zh1peng/python_CPM/blob/master/python_CPM.py)
initial version of the python CPM. should be very easy to use, just something like this:

```python
clf=CPM(threshold=0.1)     
y_pred = cross_val_predict(clf,X, y,cv=10,n_jobs=-1)    
scipy.stats.pearsonr(y,y_pred) 
```

