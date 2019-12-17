# Welcome to InteractionTransformer
![Version](https://img.shields.io/badge/version-0.1-blue.svg?cacheSeconds=2592000)

> Extract meaningful interactions from machine learning models to obtain machine-learning performance with statistical model interpretability.

Code accompanying the manuscript: "Don't Dismiss Logistic Regression: The Case for Sensible Extraction of Interactions in the Era of Machine Learning"  
Preprint: https://www.biorxiv.org/content/10.1101/2019.12.15.877134v1

Please see our wiki for more information on setting up and running this package: https://github.com/jlevy44/InteractionTransformer/wiki

**QUICKSTART DEMOS can be found here:**

Python: https://github.com/jlevy44/InteractionTransformer/blob/master/demos/InteractionTransformerPythonDemo.ipynb

R: https://github.com/jlevy44/InteractionTransformer/blob/master/demos/InteractionTransformerRDemo.Rmd

## Install

**Python:**
We recommend installing using anaconda (https://www.anaconda.com/distribution/).
First, install anaconda. Then, run:
```sh
conda create -n interaction_transform_environ python=3.7
conda activate interaction_transform_environ
```
Finally:
```sh
pip install interactiontransformer
```

**R**  
First, install the python pip package. Then:
```R
devtools::install_github("jlevy44/interactiontransformer")
```
Or:
```R
library(devtools)
install_github("jlevy44/interactiontransformer")
```

## Alternative Python Install Instructions

```sh
git clone https://github.com/jlevy44/InteractionTransformer
cd InteractionTransformer
pip install . # make sure conda is running
```

## Author

👤 **Joshua Levy**

* Github: [@jlevy44](https://github.com/jlevy44)
