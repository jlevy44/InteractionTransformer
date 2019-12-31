from sklearn.base import TransformerMixin
from kneed import KneeLocator
import copy
import shap
import dask
from dask.diagnostics import ProgressBar
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import patsy
from functools import reduce
from SafeTransformer import SafeTransformer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysnooper
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

class InteractionTransformer(TransformerMixin):
	"""Transformer object that will automatically extract interaction design terms from your data.

	Parameters
	----------
	untrained_model : sklearn-estimator
		Scikit-learn tree-based estimator that has not been fit to the data.
	max_train_test_samples : int
		Number of samples to train SHAP model off of.
	mode_interaction_extract : int/str
		Options for choosing number of interactions are 'sqrt' for square root number features,
		'knee' for experimental knee method based on interaction scores, and any integer number of interactions.
	include_self_interactions : bool
		Whether to include self-interactions / quadratic terms.
	random_state : int
		Random seed for reproducibility.
	cv_splits : int
		Number of CV splits for finding top interactions.

	Attributes
	----------
	maxn : int
		Number of samples to train SHAP model off of.
	model : sklearn-estimator
		Scikit-learn tree-based estimator that has not been fit to the data.
	mode_extract : int
		Options for choosing number of interactions are 'sqrt' for square root number features,
		'knee' for experimental knee method based on interaction scores, and any integer number of interactions.
	self_interactions : type
		Whether to include self-interactions / quadratic terms.
	random_state
	cv_splits

	"""
	def __init__(self, untrained_model=BalancedRandomForestClassifier(random_state=42,n_jobs=40),
						max_train_test_samples=100,
						mode_interaction_extract='knee',
						include_self_interactions=False,
						random_state=42,
						cv_splits=5,
						cv_scoring='auc',
						dask_scheduler='processes',
						verbose=False,
						num_workers=1,
						tree_limit=None):
		self.maxn=max_train_test_samples
		self.model=untrained_model
		assert (mode_interaction_extract in ['knee','sqrt']) or isinstance(mode_interaction_extract,int)
		assert cv_scoring in ['auc', 'acc', 'f1', 'r2', 'mae']
		assert dask_scheduler in ['threading','processes']
		self.mode_extract=mode_interaction_extract
		self.self_interactions=include_self_interactions
		self.random_state=random_state
		self.cv_splits=5
		self.cv_scoring=cv_scoring
		self.scoring_fn={'auc':roc_auc_score,
						'f1':lambda y_true,y_pred: f1_score(y_true,y_pred,average='macro'),
						'mae':mean_absolute_error,
						'r2':r2_score,
						'acc':accuracy_score}
		self.dask_scheduler=dask_scheduler
		self.feature_perturbation='tree_path_dependent'
		self.verbose=verbose
		self.num_workers=num_workers
		self.tree_limit=tree_limit

	@staticmethod
	def return_cv_score(X_train, X_test, y_train, y_test, tmp_model, scoring_fn):
		tmp_model=tmp_model.fit(X_train,y_train)
		if 'predict_proba' in dir(tmp_model):
			y_pred=tmp_model.predict_proba(X_test)
			if scoring_fn==roc_auc_score:
				y_pred=y_pred[:,-1]
				predict_mode='binary'
			else:
				y_pred=np.argmax(y_pred,axis=1)
				predict_mode='multiclass'
		else:
			y_pred=tmp_model.predict(X_test)
			predict_mode='regression'
		return scoring_fn(y_test,y_pred)

	def fit(self, X, y):
		"""Generate design matrix acquired from using SHAP on tree model.

		Parameters
		----------
		X : pd.DataFrame
			Predictors in the form of a dataframe.
		y : pd.DataFrame
			One column dataframe containing outcomes.

		Returns
		-------
		self
			Transformer with design terms.

		"""
		cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
		splits=[(X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]) for train, test in cv.split(X,y)]
		scores=[]
		for i,(X_train, X_test, y_train, y_test) in enumerate(splits):
			scores.append(dask.delayed(InteractionTransformer.return_cv_score)(X_train, X_test, y_train, y_test,copy.deepcopy(self.model),self.scoring_fn[self.cv_scoring]))
		with ProgressBar() if self.verbose else nullcontext():
			scores=dask.compute(*scores,scheduler=('processes' if self.num_workers>1 else 'single-threaded'),num_workers=self.num_workers)
		X_train, X_test, y_train, y_test = splits[np.argmax(np.array(scores))]
		model=copy.deepcopy(self.model).fit(X_train,y_train)
		if self.maxn<X_train.shape[0]-1:
			X_train,_,y_train,_=train_test_split(X_train,y_train,random_state=self.random_state,stratify=y_train,shuffle=True,train_size=self.maxn)
		if self.maxn<X_test.shape[0]-1:
			X_test,_,y_test,_=train_test_split(X_test,y_test,random_state=self.random_state,stratify=y_test,shuffle=True,train_size=self.maxn)
		explainer = shap.TreeExplainer(model, X_train, feature_perturbation=self.feature_perturbation)
		features=list(X_train)
		self.features=features
		to_sum=lambda x: x.sum(0)[0] if 'predict_proba' in dir(model) else x
		with ProgressBar() if self.verbose else nullcontext():
			shap_vals=dask.compute(*[dask.delayed(lambda x: to_sum(np.abs(explainer.shap_interaction_values(x,tree_limit=self.tree_limit))))(pd.DataFrame(X_test.iloc[i,:]).T) for i in range(X_test.shape[0])],scheduler=self.dask_scheduler,num_workers=self.num_workers)
		# import pickle
		# pickle.dump(shap_vals,open('shap_test.pkl','wb'))
		true_top_interactions=self.get_top_interactions(shap_vals)
		#print(true_top_interactions)
		self.design_terms='+'.join((np.core.defchararray.add(np.vectorize(lambda x: "Q('{}')*".format(x))(true_top_interactions.iloc[:,0]),np.vectorize(lambda x: "Q('{}')".format(x))(true_top_interactions.iloc[:,1]))).tolist())
		return self

	def get_top_interactions(self,shap_vals):
		"""Given set of SHAP interaction values extracted from model, extract top interactions.

		Parameters
		----------
		shap_vals : list
			List of SHAP values for each testing instance.

		Returns
		-------
		pd.DataFrame
			Top interactions.

		"""
		interaction_matrix=pd.DataFrame(reduce(lambda x,y:x+y,shap_vals)/len(shap_vals),columns=self.features,index=self.features)
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
		elif isinstance(self.mode_extract,int):
			n_top_interactions=self.mode_extract
		self.interaction_matrix=interation_matrix_self_interact_removed
		self.all_interaction_shap_scores=self.interaction_matrix.where(np.triu(np.ones(self.interaction_matrix.shape),k=1 if not self.self_interactions else 0).astype(np.bool)).stack().reset_index()
		self.all_interaction_shap_scores.columns=['feature_1','feature_2', 'shap_interaction_score']
		self.all_interaction_shap_scores=self.all_interaction_shap_scores.sort_values('shap_interaction_score',ascending=False)
		# top_overall_interactions=np.unravel_index(np.argsort(interation_matrix_self_interact_removed.values.ravel())[-n_top_interactions:], interaction_matrix.shape)
		# top_overall_interactions=[tuple(sorted([self.features[i],self.features[j]]))+(round(interation_matrix_self_interact_removed.iloc[i,j],6),) for i,j in np.array(top_overall_interactions).T.tolist()]
		# true_top_interactions=pd.DataFrame(top_overall_interactions).drop_duplicates()
		return self.all_interaction_shap_scores.iloc[:n_top_interactions,:]#true_top_interactions

	def refit(self, new_number_interactions=0):
		"""Modify the chosen design terms using a new number of top interactions extracted.

		Parameters
		----------
		new_number_interactions : int
			Number of new interactions to extract.

		Returns
		-------
		Transformer
			Returns a transformer object for modifying design matrices.
		"""
		if new_number_interactions:
			true_top_interactions=self.all_interaction_shap_scores.iloc[:new_number_interactions,:]
			self.design_terms='+'.join((np.core.defchararray.add(np.vectorize(lambda x: "Q('{}')*".format(x))(true_top_interactions.iloc[:,0]),np.vectorize(lambda x: "Q('{}')".format(x))(true_top_interactions.iloc[:,1]))).tolist())
		return self

	def transform(self, X):
		"""Using transformer, transform input design matrix.

		Parameters
		----------
		X : pd.DataFrame
			DataFrame containing predictors.

		Returns
		-------
		pd.DataFrame
			Transformed design matrix containing interaction terms.

		"""
		design_matrix=patsy.dmatrix(self.design_terms, data=X, return_type='dataframe')
		design_interaction_matrix=design_matrix[[col for col in list(design_matrix) if ':' in col]]
		design_interaction_matrix.columns=np.vectorize(lambda x: x.replace("Q('",'').replace("')",""))(design_interaction_matrix.columns)
		X=pd.concat([X,design_interaction_matrix],axis=1)
		self.features=list(X)
		return X

class InteractionTransformerExtraction(TransformerMixin):# one application is an iteration
	"""Experimental class that features finding nonlinear variable transformations in addition to interactions. Combines InteractionTransformeer and SAFETransformer.
	Under development"""
	def __init__(self, iterations=1, transform_first=False, untrained_model=BalancedRandomForestClassifier(random_state=42,n_jobs=40), max_train_test_samples=100, mode_interaction_extract='knee', include_self_interactions=False, penalty=3, pelt_model='l2', no_changepoint_strategy='median'):
		"""https://github.com/ModelOriented/SAFE/blob/master/SafeTransformer/SafeTransformer.py"""
		steps=[]
		for i in range(iterations):
			steps.extend([['interaction{}'.format(i),InteractionTransformer(copy.deepcopy(untrained_model), max_train_test_samples, mode_interaction_extract, include_self_interactions)],
						  ['transformer{}'.format(i),SafeTransformer(penalty=penalty, model=copy.deepcopy(untrained_model), pelt_model=pelt_model, no_changepoint_strategy=no_changepoint_strategy)]])
		self.pipeline=Pipeline(steps)

	def fit(self,X,y=None):
		self.pipeline.fit(X,y)
		return self

	def transform(self,X):
		return self.pipeline.transform(X)

def run_shap(X_train, X_test, model, model_type='tree', explainer_options={}, get_shap_values_options={}, overall=False, savefile=''):
	"""Executes a SHAP routine over training and testing design matrices given the specified model and generates useful plots.
	Returns a SHAP explainer object and the SHAP values for the model fit.

	Parameters
	----------
	X_train : pd.DataFrame
		Training design matrix.
	X_test : pd.DataFrame
		Testing design matrix.
	model : sklearn-estimator
		Fit model.
	model_type : str
		SHAP model type, choose between tree, kernel and linear.
	explainer_options : dict
		Additional custom options for finding explainer object.
	get_shap_values_options : dict
		Additional custom options for finding SHAP values.
	overall : bool
		Return bar chart of overall SHAP scores, top features or dot plot of score breakdown.
	savefile : str
		Save file for SHAP output image.

	Returns
	-------
	explainer
		SHAP explainer object
	shap_values
		np.array or list containing SHAP scores.

	"""

	shap_model={'tree':shap.TreeExplainer,'kernel':shap.KernelExplainer,'linear':shap.LinearExplainer}[model_type]

	explainer = shap_model(model, X_train,**explainer_options)

	shap_values = explainer.shap_values(X_test,**get_shap_values_options)

	if model_type=='tree' and model.__class__.__name__!='XGBClassifier':
		shap_values=np.array(shap_values)[1,...]

	plt.figure()
	shap.summary_plot(shap_values, X_test,feature_names=list(X_train), plot_type='bar' if overall else 'dot', max_display=30)
	if savefile:
		plt.savefig(savefile,dpi=300)
	return explainer, shap_values
