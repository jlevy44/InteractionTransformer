library(reticulate)

#' Install InteractionTransformer Python Package.
#'
#' This function installs the interaction transformer package from PyPI and places it in local or conda environment.
#'
#' @param custom.path Location/name of pip package.
#' @param pip Whether to use PyPI.
#' @param conda.env Name of conda environment to save to.
#' @export
install_transformer <- function(custom.path='git+https://github.com/jlevy44/InteractionTransformer',pip=F, conda.env='interactiontransformer') {
  reticulate:::py_install(custom.path,pip=pip,envname=conda.env)
}

#' Create conda environment with specified name.
#'
#' @param conda.env Name of conda environment.
#' @export
create.conda <- function(conda.env='interactiontransformer'){
  reticulate:::create_conda(conda.env)
}

#' Search for existing conda environment.
#'
#' @param conda.env Name of conda environment.
#' @export
detect.conda <- function(conda.env='interactiontransformer'){
  reticulate:::py_install_method_detect(conda.env)
}

#' Source python from user specified path.
#'
#' @param python.exec Python executable.
#' @export
source.python <- function(python.exec='/usr/local/bin/python3'){
  reticulate:::use_python(python.exec)
}

#' Import interaction transformer package after sourcing python.
#'
#' @export
import_transformer <- function() {
  interactiontransformer<-reticulate:::import('interactiontransformer')
}

#' Split data into training and testing datasets. Y must be a single column dataframe.
#'
#' @param X Predictors in the form of a dataframe.
#' @param y One column dataframe containing outcomes.
#' @param train_size Fraction of dataset to split by, retains this fraction as training set.
#' @return A list containing training and testing design matrices and outcome data.
#' @export
train.test.split<- function(X,y,train_size=0.8) {
  train_test_splits<-interactiontransformer$InteractionTransformer$train_test_split(X,y,stratify=y,train_size=train_size,random_state=42L,shuffle=TRUE)
  train_test_splits<-list(X.train=train_test_splits[1][[1]],X.test=train_test_splits[2][[1]],y.train=train_test_splits[3][[1]],y.test=train_test_splits[4][[1]])
  return(train_test_splits)
}

#' Generate design matrix acquired from using SHAP on tree model.
#'
#' Returns transformer object. Y must be a single column dataframe.
#'
#' @param X.train Predictors in the form of a dataframe.
#' @param y.train One column dataframe containing outcomes.
#' @param untrained_model Scikit-learn tree-based estimator that has not been fit to the data.
#' @param max_train_test_samples Number of samples to train SHAP model off of.
#' @param mode_interaction_extract : Options for choosing number of interactions are 'sqrt' for square root number features, 'knee' for experimental knee method based on interaction scores, and any integer number of interactions.
#' @param include_self_interactions Whether to include self-interactions / quadratic terms.
#' @param cv_splits Number of CV splits for finding top interactions.
#' @return Interaction transformer object that can be applied to the design matrix.
#' @export
interaction.fit <- function(X.train,y.train, untrained_model='default', max_train_test_samples=100, mode_interaction_extract='knee', include_self_interactions=F, cv_splits=5L) {
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
                                                                                      cv_splits=cv_splits)
  transformer<-transformer$fit(X.train,y.train)
  return(transformer)
}

#' Modify the chosen design terms using a new number of top interactions extracted.
#' @param new_number_interactions Number of new interactions to extract.
#' @return Interaction transformer object that can be applied to the design matrix.
#' @export
refit.transformer <- function(transformer, new_number_interactions=0L) {
  return(transformer$refit(new_number_interactions))
}

#' Using transformer, transform input design matrix.
#' @param transformer Interaction Transformer object.
#' @param X Input design matrix.
#' @return Output design matrix with interaction terms.
#' @export
interaction.transform <- function(transformer, X) {
  return(transformer$transform(X))
}

#' Executes a SHAP routine over training and testing design matrices given the specified model and generates useful plots.
#' @param X.train Training design matrix.
#' @param X.test Testing design matrix.
#' @param model Fit model.
#' @param model_type SHAP model type, choose between tree, kernel and linear.
#' @param savefile Save file for SHAP output image.
#' @export
run.shap <- function(X.train, X.test, model, model_type='tree', savefile='shap.output.png') {
  interactiontransformer$InteractionTransformer$run_shap(X.train, X.test, model, model_type=model_type,savefile=savefile)
}

#' Given set of SHAP interaction values extracted from model, extract top interactions.
#'
#' @param shap_vals List of SHAP values for each testing instance.
#' @return DataFrame containing top interactions.
#' @export
get.top.interactions <- function(transformer) {
  return(transformer$all_interaction_shap_scores)
}
