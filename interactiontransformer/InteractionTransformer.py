from sklearn.base import TransformerMixin
from kneed import KneeLocator # git clone and pip install
import copy
import shap
import dask
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import patsy
from functools import reduce
from SafeTransformer import SafeTransformer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class InteractionTransformer(TransformerMixin):
    def __init__(self, untrained_model=BalancedRandomForestClassifier(random_state=42,n_jobs=40), max_train_test_samples=100, mode_interaction_extract='knee', include_self_interactions=False):
        self.maxn=max_train_test_samples
        self.model=untrained_model
        assert (mode_interaction_extract in ['knee','sqrt']) or isinstance(mode_interaction_extract,int)
        self.mode_extract=mode_interaction_extract
        self.self_interactions=include_self_interactions


    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits=[(X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]) for train, test in cv.split(X,y)]
        scores=[]
        for i,(X_train, X_test, y_train, y_test) in enumerate(splits):
            y_probs=copy.deepcopy(self.model).fit(X_train,y_train).predict_proba(X_test)[:,-1]
            scores.append(roc_auc_score(y_test,y_probs))
        X_train, X_test, y_train, y_test = splits[np.argmax(np.array(scores))]
        model=copy.deepcopy(self.model).fit(X_train,y_train)
        if self.maxn<X_train.shape[0]:
            X_train,_,y_train,_=train_test_split(X_train,y_train,random_state=42,stratify=y_train,shuffle=True,train_size=self.maxn)
        if self.maxn<X_test.shape[0]:
            X_test,_,y_test,_=train_test_split(X_test,y_test,random_state=42,stratify=y_test,shuffle=True,train_size=self.maxn)
        explainer = shap.TreeExplainer(model, X_train)
        features=list(X_train)
        self.features=features
        shap_vals=dask.compute(*[dask.delayed(lambda x: np.abs(explainer.shap_interaction_values(x)).sum(0))(X_test.iloc[i,:]) for i in range(X_test.shape[0])],scheduler='processes')
        true_top_interactions=self.get_top_interactions(shap_vals)
        self.design_terms='+'.join((np.core.defchararray.add(np.vectorize(lambda x: "Q('{}')*".format(x))(true_top_interactions[0]),np.vectorize(lambda x: "Q('{}')".format(x))(true_top_interactions[1]))).tolist())
        return self

    def get_top_interactions(self,shap_vals): # add knee locator
        interaction_matrix=pd.DataFrame(reduce(lambda x,y:x+y,shap_vals)/len(shap_vals),columns=self.features,index=self.features)
        # else:
        #     n_top_interactions=int(fraction_interactions*interaction_matrix.shape[0]*(interaction_matrix.shape[0]-1)/2)
        interation_matrix_self_interact_removed=interaction_matrix.copy()
        if not self.self_interactions:
            for i in np.arange(interaction_matrix.shape[0]):
                interation_matrix_self_interact_removed.iloc[i,i]=0
        if self.mode_extract=='knee':
            try:
                x,y=np.arange(interation_matrix_self_interact_removed.shape[0]**2),np.sort(interation_matrix_self_interact_removed.values.ravel())[::-1]
                kneed=KneeLocator(x, y, direction='decreasing', curve='convex')
                n_top_interactions=min(100,kneed.knee)
            except Exception as e:
                print(e)
                print('Error Detected: Defaulting to SQRT calculation of number of new interaction features.')
                self.mode_extract='sqrt'
        if self.mode_extract=='sqrt':
            n_top_interactions=int(np.sqrt(interaction_matrix.shape[0]))
        else:
            n_top_interactions=self.mode_extract
        self.interaction_matrix=interation_matrix_self_interact_removed
        top_overall_interactions=np.unravel_index(np.argsort(interation_matrix_self_interact_removed.values.ravel())[-n_top_interactions:], interaction_matrix.shape)
        top_overall_interactions=[tuple(sorted([self.features[i],self.features[j]]))+(round(interation_matrix_self_interact_removed.iloc[i,j],6),) for i,j in np.array(top_overall_interactions).T.tolist()]
        true_top_interactions=pd.DataFrame(top_overall_interactions).drop_duplicates()
        return true_top_interactions

    def transform(self, X):
        design_matrix=patsy.dmatrix(self.design_terms, data=X, return_type='dataframe')
        design_interaction_matrix=design_matrix[[col for col in list(design_matrix) if ':' in col]]
        design_interaction_matrix.columns=np.vectorize(lambda x: x.replace("Q('",'').replace("')",""))(design_interaction_matrix.columns)
        X=pd.concat([X,design_interaction_matrix],axis=1)
        self.features=list(X)
        return X

class InteractionTransformerExtraction(TransformerMixin):# one application is an iteration
    def __init__(self, iterations=1, transform_first=False, untrained_model=BalancedRandomForestClassifier(random_state=42,n_jobs=40), max_train_test_samples=100, mode_interaction_extract='knee', include_self_interactions=False, penalty=3, pelt_model='l2', no_changepoint_strategy='median'):
        """https://github.com/ModelOriented/SAFE/blob/master/SafeTransformer/SafeTransformer.py"""
        self.interaction=InteractionTransformer(copy.deepcopy(untrained_model), max_train_test_samples, mode_interaction_extract, include_self_interactions)
        self.transformation=SafeTransformer(penalty=penalty, model=copy.deepcopy(untrained_model), pelt_model=pelt_model, no_changepoint_strategy=no_changepoint_strategy)
        self.pipeline=Pipeline(list(reduce(lambda x,y:x+y,[[('interaction{}'.format(i),copy.deepcopy(self.interaction)),('transformation{}'.format(i),copy.deepcopy(self.transformation))][::(-1 if transform_first else 1)] for i in range(iterations)])))

    def fit(self,X,y=None):
        self.pipeline.fit(X,y)
        return self

    def transform(self,X):
        return self.pipeline.transform(X)
