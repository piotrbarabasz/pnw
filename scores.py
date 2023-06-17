import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel

scores = np.load("scores.npy")

table = tabulate(scores,
    tablefmt="grid",
    headers=[
        "split 1",
        "split 2",
        "split 3",
        "split 4",
        "split 5"
    ],
    showindex=[
         'tfidf_BR_MNB',
         # 'tfidf_BR_LR',
         'tfidf_CC_MNB',
         # 'tfidf_CC_LR',
         'tfidf_LB_MNB',
         # 'tfidf_LB_LR',
         # 'cv_BR_MNB',
         # 'cv_BR_LR',
         'cv_CC_MNB',
         # 'cv_CC_LR',
         # 'cv_LP_MNB',
         # 'cv_LP_LR',
         # 'cv_LP_SVC_lin'
    ])

print(table)
result = []

for classifier_idx in range(scores.shape[0]):
    classifier_result = []
    for compare_classifier_idx in range(scores.shape[0]):
        classifier_result.append(ttest_rel(
            scores[classifier_idx, :],
            scores[compare_classifier_idx, :]
        ))
    result.append(classifier_result)

result_array = np.array(result)

table_statistic = tabulate((result_array[:, :, 0]),
    tablefmt="grid",
    headers=[
       'tfidf_BR_MNB',
       # 'tfidf_BR_LR',
       'tfidf_CC_MNB',
       # 'tfidf_CC_LR',
       'tfidf_LB_MNB',
       # 'tfidf_LB_LR',
       # 'cv_BR_MNB',
       # 'cv_BR_LR',
       'cv_CC_MNB',
       # 'cv_CC_LR',
       # 'cv_LP_MNB',
       # 'cv_LP_LR',
       # 'cv_LP_SVC_lin'
    ],
    showindex=[
       'tfidf_BR_MNB',
       # 'tfidf_BR_LR',
       'tfidf_CC_MNB',
       # 'tfidf_CC_LR',
       'tfidf_LB_MNB',
       # 'tfidf_LB_LR',
       # 'cv_BR_MNB',
       # 'cv_BR_LR',
       'cv_CC_MNB',
       # 'cv_CC_LR',
       # 'cv_LP_MNB',
       # 'cv_LP_LR',
       # 'cv_LP_SVC_lin'
    ])

table_pvalue = tabulate((result_array[:, :, 1] < 0.05).astype(int),
    tablefmt="grid",
    headers=[
        'tfidf_BR_MNB',
        # 'tfidf_BR_LR',
        'tfidf_CC_MNB',
        # 'tfidf_CC_LR',
        'tfidf_LB_MNB',
        # 'tfidf_LB_LR',
        # 'cv_BR_MNB',
        # 'cv_BR_LR',
        'cv_CC_MNB',
        # 'cv_CC_LR',
        # 'cv_LP_MNB',
        # 'cv_LP_LR',
        # 'cv_LP_SVC_lin'
    ],
    showindex=[
        'tfidf_BR_MNB',
        # 'tfidf_BR_LR',
        'tfidf_CC_MNB',
        # 'tfidf_CC_LR',
        'tfidf_LB_MNB',
        # 'tfidf_LB_LR',
        # 'cv_BR_MNB',
        # 'cv_BR_LR',
        'cv_CC_MNB',
        # 'cv_CC_LR',
        # 'cv_LP_MNB',
        # 'cv_LP_LR',
        # 'cv_LP_SVC_lin'
    ])

print('Results of statistic')
print(table_statistic)
print('Results of p value')
print(table_pvalue)
