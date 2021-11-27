# COICOP text classification

## Intro

This is the code repository related to a project that revolved around classifying product descriptions into COICOP2018 categories. Various kinds of automatic classifiers where developed and tested through the course of the project.

the abreiviated results where as follows:
```
--------------------------------------------------------------------------------
               percision    recall    f1-score  support
XLMR-enis-26

    accuracy                           0.93      2059
   macro avg       0.86      0.84      0.84      2059
weighted avg       0.93      0.93      0.93      2059
--------------------------------------------------------------------------------
iceBERT-26

    accuracy                           0.91      2059
   macro avg       0.81      0.80      0.79      2059
weighted avg       0.91      0.91      0.91      2059
--------------------------------------------------------------------------------
fasttext

    accuracy                           0.91      2059
   macro avg       0.80      0.79      0.79      2059
weighted avg       0.91      0.91      0.91      2059
--------------------------------------------------------------------------------
isl-fasttext

    accuracy                           0.87      2059
   macro avg       0.80      0.77      0.78      2059
weighted avg       0.87      0.87      0.87      2059
--------------------------------------------------------------------------------
naive-bayes

    accuracy                           0.77      2059
   macro avg       0.61      0.41      0.46      2059
weighted avg       0.77      0.77      0.74      2059
```

## General description of the content of this repository

### Training scripts

```trainBERT-like-coicop.py``` is an example script showing the neccesary steps we used in training on XLMR-ENIS and IceBERT models\
```fast_text_classifier_valid_data.py``` is the script we used to train on fasttext model.\
```train_isl_fasttext.py``` the script to train on an icelandic version of fastext\
```the naive-bayes scripts: bayes-from-train-dev-test.py. naive_bayes_coicop_v1.py and naive_bayes_with_lemmas_coicop_v2.py``` all involve training but also include use of the models


### Evaluation scripts
evalBERT-like-coicop.py\
eval_fasttext.py\
eval_isl_fasttext.py\
bayes-from-train-dev-test.py ---> with evaluation portion of code uncommented

### Use Scripts
use_enis.py\
use_fasttext.py\
use_isl_fasttext.py\
bayes-from-train-dev-text.py\
naive_bayes_with_lemmas_coicop_v2.py
