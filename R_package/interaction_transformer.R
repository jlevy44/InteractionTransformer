library(reticulate)

install_transformer <- function(custom.path='interactiontransformer',pip=T) {
  py_install(custom.path,pip=pip)
}

detect.conda <- function(conda.env='py36'){
  reticulate:::py_install_method_detect(conda.env)
}

source.python <- function(python.dir='/usr/local/bin/python3'){
  reticulate:::use_python(python.dir)
}

import_transformer <- function() {
  interactiontransformer<-import('interactiontransformer')
}

train.test.split<- function(X,y) {
  train_test_splits<-interactiontransformer$InteractionTransformer$train_test_split(X,y,stratify=y)
  train_test_splits<-list(X.train=train_test_splits[1],X.test=train_test_splits[2],y.train=train_test_splits[3],y.test=train_test_splits[4])
  return(train_test_splits)
}

interaction.fit <- function(X.train,y.train, untrained_model='default') {
  if (untrained_model=='default') {
    untrained_model<-interactiontransformer$InteractionTransformer$BalancedRandomForestClassifier(random_state = 42)
  }
  transformer <- interactiontransformer$InteractionTransformer$InteractionTransformer(untrained_model = model)
  transformer<-interaction.transform$fit(X.train,y.train)
  return(transformer)
}

interaction.transform <- function(transformer, X) {
  return(transformer$transform(X))
}

run.shap <- function(X.train, X.test, model, model_type='tree', savefile='shap.output.png') {
  interactiontransformer$InteractionTransformer$run_shap(X_train, X_test, model, model_type=model_type,savefile=savefile)
}

get.top.interactions <- function(transformer) {
  return(transformer$all_interaction_shap_scores)
}