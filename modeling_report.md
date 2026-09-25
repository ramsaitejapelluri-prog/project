# Modeling results

## Class balance and split

{'not_survived': {'count': 549, 'proportion': 0.6175478065241845}, 'survived': {'count': 340, 'proportion': 0.38245219347581555}}

Stratification matters because it preserves the observed class proportions in both train and test sets.

## Imbalance comparison

             strategy  precision   recall       f1
             baseline   0.783333 0.691176 0.734375
class_weight_balanced   0.718310 0.750000 0.733813
     SMOTE_train_only   0.735294 0.735294 0.735294

The SMOTE_train_only strategy performed best by F1 (0.735), with precision=0.735 and recall=0.735. It offers the strongest balance between false positives and false negatives on the held-out test set.

## Regression residuals

The residual plot suggests heteroscedasticity because absolute residual size changes with predicted fare.

## Model comparison

                 model  accuracy  precision   recall       f1      auc     confusion_matrix       MAE      RMSE       R2  Adjusted_R2
   Logistic Regression  0.808989   0.783333 0.691176 0.734375 0.860963 [[97, 13], [21, 47]]       NaN       NaN      NaN          NaN
         Decision Tree  0.764045   0.760000 0.558824 0.644068 0.837366 [[98, 12], [30, 38]]       NaN       NaN      NaN          NaN
         Random Forest  0.803371   0.761905 0.705882 0.732824 0.823663 [[95, 15], [20, 48]]       NaN       NaN      NaN          NaN
Fare Linear Regression       NaN        NaN      NaN      NaN      NaN                  NaN 21.138552 41.746502 0.346774      0.31178

The recommended deployment classifier is Tuned Random Forest, with F1=0.785, AUC=0.829, precision=0.823, and recall=0.750. It provides the strongest combined held-out performance and is saved with its complete preprocessing pipeline. The precision and recall trade-off should still be reviewed against the operational cost of missed survivors. Fare regression is a separate task and is not ranked against the classification metrics.
