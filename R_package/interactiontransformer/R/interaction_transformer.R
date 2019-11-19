library(reticulate)

install_transformer <- function(custom.path='interactiontransformer',pip=F, conda.env='interactiontransformer') {
  reticulate:::py_install(custom.path,pip=pip,envname=conda.env)
}

create.conda <- function(conda.env='interactiontransformer'){
  reticulate:::create_conda(conda.env)
}

detect.conda <- function(conda.env='interactiontransformer'){
  reticulate:::py_install_method_detect(conda.env)
}

source.python <- function(python.dir='/usr/local/bin/python3'){
  reticulate:::use_python(python.dir)
}

import_transformer <- function() {
  interactiontransformer<-reticulate:::import('interactiontransformer')
}

train.test.split<- function(X,y,train_size=0.8) {
  train_test_splits<-interactiontransformer$InteractionTransformer$train_test_split(X,y,stratify=y,train_size=train_size,random_state=42L,shuffle=TRUE)
  train_test_splits<-list(X.train=train_test_splits[1][[1]],X.test=train_test_splits[2][[1]],y.train=train_test_splits[3][[1]],y.test=train_test_splits[4][[1]])
  return(train_test_splits)
}

interaction.fit <- function(X.train,y.train, untrained_model='default', max_train_test_samples=100, mode_interaction_extract='knee', include_self_interactions=F) {
  if (untrained_model=='default') {
    untrained_model<-interactiontransformer$InteractionTransformer$BalancedRandomForestClassifier(random_state = 42L)
  }
  if (is.numeric(mode_interaction_extract)) {
    mode_interaction_extract<-as.integer(mode_interaction_extract)
  }
  y.train<-as.data.frame(y.train)
  transformer <- interactiontransformer$InteractionTransformer$InteractionTransformer(untrained_model = untrained_model,
                                                                                      max_train_test_samples=max_train_test_samples,
                                                                                      mode_interaction_extract=mode_interaction_extract,
                                                                                      include_self_interactions=include_self_interactions,
                                                                                      random_state=42L,
                                                                                      cv_splits=5L)
  transformer<-transformer$fit(X.train,y.train)
  return(transformer)
}

interaction.transform <- function(transformer, X) {
  return(transformer$transform(X))
}

run.shap <- function(X.train, X.test, model, model_type='tree', savefile='shap.output.png') {
  interactiontransformer$InteractionTransformer$run_shap(X.train, X.test, model, model_type=model_type,savefile=savefile)
}

get.top.interactions <- function(transformer) {
  return(transformer$all_interaction_shap_scores)
}
