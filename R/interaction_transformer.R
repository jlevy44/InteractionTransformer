library(reticulate)
install_transformer <- function(custom.path='interactiontransformer',pip=T) {
  py_install(custom.path,pip=pip)
}
detect.conda <- function(conda.env='py36'){
  reticulate:::py_install_method_detect(conda.env)
}

interactiontransformer<-import('interactiontransformer')

interaction.fit <- function(X.train,y.train, untrained_model=interactiontransformer$InteractionTransformer$BalancedRandomForestClassifier()) {
  transformer <- interactiontransformer$InteractionTransformer$InteractionTransformer(untrained_model = model)
  transformer<-interaction.transform$fit(X.train,y.train)
  return(transformer)
}

interaction.transform <- function(transformer, X) {
  return(transformer$transform(X))
}

run.shap <- function(X.train, X.test, model, model_type='tree', savefile='shap.output.png') {
  interactiontransformer$InteractionTransformer$run_shap(run_shap(X_train, X_test, model, model_type=model_type,savefile=savefile)
}

get.top.interactions <- function(transformer) {
  return(transformer$all_interaction_shap_scores)
}