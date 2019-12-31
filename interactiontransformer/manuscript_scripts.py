from interactiontransformer.InteractionTransformer import InteractionTransformer
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os
import matplotlib
from interactiontransformer.utils import get_dataset, preprocess_data
matplotlib.use('Agg')

def interaction_extraction_pipeline(datasets_dir='datasets',ID=0,results_df=pd.read_csv('results.csv')):
    X,y,cat=get_dataset('{}/{}.p'.format(datasets_dir,ID))

    X,y=next(preprocess_data(X,y,cat,return_xy=True))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores=[]

    for train, test in cv.split(X,y):
        X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]
        transformer=InteractionTransformer(max_train_test_samples=1000,mode_interaction_extract=int(2*np.sqrt(X.shape[1])))
        transformer.fit(X_train,y_train)
        X_train=transformer.transform(X_train)
        X_test=transformer.transform(X_test)
        model=LogisticRegression(class_weight='balanced',random_state=42)
        y_probs=model.fit(X_train,y_train).predict_proba(X_test)[:,-1]
        score=roc_auc_score(y_test,y_probs)
        scores.append(score)
    scores=np.mean(scores)
    return scores

def run_preliminary_models(preprocess_gen=()):
    scores=dict(lr=[],rf=[])
    try:
        opts=dict(random_state=42)
        for X_train, X_test, y_train, y_test, info_dict in preprocess_gen:
            models=dict(lr=LogisticRegression(class_weight='balanced',**opts),rf=BalancedRandomForestClassifier(**opts))
            for model in list(models.keys()):
                y_probs=models[model].fit(X_train,y_train).predict_proba(X_test)[:,-1]
                scores[model].append(roc_auc_score(y_test,y_probs))
        for score in list(scores.keys()):
            if score=='rf':
                scores['top_split']=np.argmax(scores[score])
            scores[score]=np.mean(scores[score])
        scores.update(info_dict)
    except Exception as e:
        print(e)
        for score in list(scores.keys()):
            scores[score]=-1
        for k in ['n','p','pcat','cb','top_split']:
            scores[k]=-1
    return scores

def preliminary_analysis(dataset_path='datasets',result_csv='results_new.csv'):
    datasets={int(f.split('/')[-1][:-2]):f for f in glob.glob('{}/*.p'.format(dataset_path))}
    scores=[dask.delayed(lambda x: run_models(preprocess_data(*get_dataset(x))))(datasets[i]) for i in list(datasets.keys())]
    scores=dask.compute(*scores,scheduler='processes',num_workers=15)
    pd.DataFrame(scores,index=list(datasets.keys())).to_csv(result_csv)

def add_interaction_results(results_csv='results_new.csv',final_results_pkl='results_interactions.pkl'):
    results_df=pd.read_csv(results_csv,index_col=0)
    results_df=results_df[results_df['lr']!=-1].dropna()
    results_df=results_df[(results_df['p']<=110) & (results_df['p']>=5)& (results_df['n']>=50) & (results_df['n']<2500)]
    scores={}
    if os.path.exists(final_results_pkl):
        scores=pickle.load(open(final_results_pkl,'rb'))
    for i in results_df.index.values:
        if i not in list(scores.keys()):
            score=interaction_extraction_pipeline(i,results_df)
            scores[i]=score
            pickle.dump(scores,open(final_results_pkl,'wb'))
