import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys
import os
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from timeit import default_timer as timer
from sklearn.model_selection import permutation_test_score
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from scipy import interp
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from lightgbm import LGBMClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


class curve_and_metrics:
    """Class for gathering metrics and curve plotting"""

    def __init__(self, loops):

        # Variables for precision-recall curves
        self.pooled_y = []
        self.pooled_prob = []

        # Variables for importance gathering
        self.outer_importances = []
        self.inner_importances = []

        # Mean false positive rate for roc curves
        self.mean_fpr = np.linspace(0, 1, 100)
        self.inner_tprs = []
        self.inner_aucs = []

        # Outer
        self.outer_tprs = []
        self.outer_aucs = []

        self.p_list = []

    def inner_loop(self, y_test, y_prob, importances):
        """Add values from each nested cv outer fold (meaning innner)
        outer means the iteration averaged values of a nested cv"""

        self.inner_importances.append(importances)
        self.pooled_y.append(y_test)
        self.pooled_prob.append(y_prob)

        # Generate roc curves and auc values
        real_tpr, real_auc = self.tpr_and_auc(y_test, y_prob)

        # Add real values
        self.inner_tprs.append(real_tpr)
        self.inner_aucs.append(real_auc)

    def tpr_and_auc(self, y_test, y_prob):
        """Returns the AUC value and roc curve based on y_test and y_prob"""
        # Roc curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Interpolate 100 points from fpr and tpr
        interp_tpr = interp(self.mean_fpr, fpr, tpr)
        # Set the first point to zero, so it will look better
        interp_tpr[0] = 0.0
        roc_auc = (auc(fpr, tpr))

        return interp_tpr, roc_auc

    def end_inner(self):
        """Average the iteration tprs, auc's and importances"""
        self.outer_importances.append(np.mean(np.array(self.inner_importances), axis=0))

        mean_tpr = np.mean(self.inner_tprs, axis=0)
        mean_tpr[-1] = 1.0

        # Add real values
        self.outer_tprs.append(mean_tpr)
        self.outer_aucs.append(np.mean(self.inner_aucs))

        # Reset the inner lists
        self.inner_importances = []
        self.inner_tprs = []
        self.inner_aucs = []

    def permutation_p(self, permutation_scores):
        """Compare the nested cv averaged auc to the permutation scores and calculate p-value,
        from https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_validation.py#L984"""

        self.p_list.append((np.sum(permutation_scores >= self.outer_aucs[-1]) + 1.0) / (len(permutation_scores) + 1))

    def create_final(self, file_name):
        """Create the final files with curves, upper and lower bounds, p-values and more"""

        with open(f'{file_name}_report.txt', 'w+') as writefile:

            # Initialize data frame to save all the information for reproducing the graph
            roc_curve_df = pd.DataFrame(data={'False positive rate': self.mean_fpr})

            mean_tpr = np.mean(self.outer_tprs, axis=0)
            mean_auc = auc(self.mean_fpr, mean_tpr)
            std_auc = np.std(self.outer_aucs)

            writefile.write(f'Real classifier ROC_AUC: {mean_auc}\n'
                            f'Real classifier ROC_AUC_STD: {std_auc}\n')

            std_tpr = np.std(self.outer_tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            roc_curve_df['True positive rate'] = mean_tpr
            roc_curve_df['Lower confidence interval'] = tprs_lower
            roc_curve_df['Upper confidence interval'] = tprs_upper

            roc_curve_df.to_csv(f'{file_name}_roc_curve.tsv', index=False, sep='\t')


            # Make the precision-recall curve
            y_real = np.concatenate(self.pooled_y)
            y_proba = np.concatenate(self.pooled_prob)

            # Real classifier
            precision, recall, _ = precision_recall_curve(y_real, y_proba)
            pr_curve_df = pd.DataFrame(data={'precision': precision,
                                             'recall': recall})
            pr_auc = average_precision_score(y_real, y_proba)

            pr_curve_df.to_csv(f'{file_name}_pr_curve.tsv', index=False, sep='\t')

            writefile.write(f'Real classifier PR_AUC: {pr_auc}\n')

            fisher = stats.combine_pvalues(self.p_list, method='fisher')
            writefile.write(f'Fisher combined p-value: {fisher[1]}\n'
                            f'Chi-squared statistic:: {fisher[0]}\n')

    def create_importances(self, file_name, importance_names):

        # For rank aggregation create a list with each importance
        rank_csv = pd.DataFrame(self.outer_importances, columns=importance_names)
        rank_csv.to_csv(f'{file_name}_importances_ranks.tsv', index=False, sep='\t')

        # Importances
        importances_mean = np.mean(self.outer_importances, axis=0)
        importances_std = np.std(self.outer_importances, axis=0)

        sorted_mean_importances = np.argsort(importances_mean)
        sorted_mean_importances = np.flip(sorted_mean_importances, axis=0)

        # Save the importances and OTU ID's
        importance_csv = pd.DataFrame([np.take(importances_mean, sorted_mean_importances),
                                       np.take(importances_std, sorted_mean_importances)],
                                      columns=np.take(importance_names, sorted_mean_importances))
        importance_csv.to_csv(f'{file_name}_importances_names.tsv', index=False, sep='\t')


def make_X_similar(x1_col_names, x2, x2_col_names):
    """Returns X_train and X_col_names in the same shape and order as x1"""

    df2 = pd.DataFrame(data=x2, columns=x2_col_names)

    # Take feature found in df1
    common_features = []
    for feature in x1_col_names:
        if feature in df2.columns.values:
            common_features.append(feature)
    new_df2 = df2[common_features]

    # create the features with zeros in new_df2
    for feature in x1_col_names:
        if feature not in new_df2.columns.values:
            new_df2[feature] = [0 for i in range(df2.shape[0])]

    # Make sure the order is the same as in x1
    new_df2 = new_df2[x1_col_names]

    return new_df2.values, new_df2.columns.values


class VifSelector(BaseEstimator, TransformerMixin):

    def __init__(self,
                 dataset_name=None,
                 random_state=None):

        self.random_state = random_state
        self.feature_indeces = None
        self.dataset_name = dataset_name

    def fit(self, X, y=None):

        if self.random_state:
            feature_ind = self.random_state.randint(0, 100)
        else:
            feature_ind = np.random.randint(0, 100)

        i_list = []
        with open(f'{self.dataset_name}_vifs.txt', 'r') as file:
            for line in file:
                i_list.append(line.rstrip('\n').split(','))

        i_list = i_list[feature_ind]

        i_list = [int(i) for i in i_list]

        self.feature_indeces = i_list

        return self

    def transform(self, X, y=None):

        return np.take(X, self.feature_indeces, axis=1)


class PipelineSelector(BaseEstimator, TransformerMixin):

    def __init__(self,
                 column_names,
                 clustering_method=None,
                 tax_database=None,
                 feature_table_type=None,
                 relative=None,
                 prevalence_threshold=None):

        self.column_names = column_names
        self.clustering_method = clustering_method
        self.tax_database = tax_database
        self.feature_table_type = feature_table_type
        self.relative = relative
        self.prevalence_threshold = prevalence_threshold

        self.feature_indeces = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        feature_tag = f'/{self.clustering_method}/{self.tax_database}/{self.feature_table_type}/{self.relative}/prev_{self.prevalence_threshold}_ML_table.tsv/'
        tag_index = []
        for i, col in enumerate(self.column_names):
            if feature_tag in col:
                tag_index.append(i)

        self.feature_indeces = tag_index

        new_X = X.copy()
        return np.take(new_X, tag_index, axis=1)


class PrefitAveragingClassifier(BaseEstimator, ClassifierMixin):
    """Estimator to take prefit estimators and output the predicted averaged probabilities"""

    def __init__(self, estimator_list):
        self.estimator_list = estimator_list

    def predict_proba(self, X):

        predicted = []
        for clf in self.estimator_list:
            predicted.append(clf.predict_proba(X))

        return np.mean(predicted, axis=0)

    def fit(self, X, y):
        return self


# Gather all the target datasets folders
month = '_month'
target_root = f'/COMBO_2021/{month}_taxa_baseline/'
dir_list = os.listdir(target_root)
target_dirs = []

for dir_name in dir_list:
    if month in dir_name:
        if os.path.isdir(f'{target_root}{dir_name}'):
            target_dirs.append(dir_name)

# Gather all the predictor dataset folders
pred_month = '_month'
pred_root = f'/COMBO_2021/{pred_month}_taxa_baseline/'
dir_list = os.listdir(pred_root)
pred_target_dirs = []

for dir_name in dir_list:
    if pred_month in dir_name:
        # make sure its a directory and not a file
        if os.path.isdir(f'{pred_root}{dir_name}'):
            pred_target_dirs.append(dir_name)

rng = np.random.RandomState(15112021)

for target_dir in target_dirs:

    # Figure out how many outer folds there are
    fold_count = int((len(os.listdir(f'{target_root}{target_dir}/rf_{target_dir}_classifiers/'))-1)/3/40)
    performance = curve_and_metrics(loops=2)

    outer_importances = []
    score_list = []
    for repeat in range(40):
        for i_fold in range(fold_count):
            # Samples with class labels
            X_re = load(f'{target_root}{target_dir}/rf_{target_dir}_classifiers/{repeat}_{i_fold}_Xtest.joblib')
            y_re = load(f'{target_root}{target_dir}/rf_{target_dir}_classifiers/{repeat}_{i_fold}_Ytest.joblib')
            X_col_names = load(f'{target_root}{target_dir}/rf_{target_dir}_classifiers/importances_column_names.joblib')

            any_pred_col_names = load(f'{pred_root}{pred_target_dirs[0]}/rf_{pred_target_dirs[0]}_classifiers/importances_column_names.joblib')
            # Make sure that the target data is same shape as what the predictors were trained with
            if month != pred_month:
                X_re, X_col_names = make_X_similar(x1_col_names=any_pred_col_names,
                                                   x2=X_re,
                                                   x2_col_names=X_col_names)

            model_list = []
            for pred_dir in pred_target_dirs:
                # Figure out which model was the best and only use that
                best_score = 0
                for alg_type in ['mlp', 'lgbm', 'rf', 'extra']:
                    with open(f'{pred_root}{pred_dir}/{alg_type}_{pred_dir}_report.txt', 'r') as file:
                        for line in file:
                            if 'Real classifier ROC_AUC:' in line:
                                words = line.rstrip('\n').split(' ')
                                real_auc = float(words[-1])
                                if real_auc > best_score:
                                    best_alg = alg_type
                                    best_score = real_auc

                if target_dir.split('_')[0] in pred_dir:
                    continue

                if os.path.isfile(f'{pred_root}{pred_dir}/{best_alg}_{pred_dir}_classifiers/{repeat}_{i_fold}_model.joblib'):
                    model_list.append(load(f'{pred_root}{pred_dir}/{best_alg}_{pred_dir}_classifiers/{repeat}_{i_fold}_model.joblib'))
                else:
                    # Predictor had fewer folds. Pick a random model from the directory instead
                    random_fold = rng.randint(0, 8)
                    model_list.append(load(f'{pred_root}{pred_dir}/{best_alg}_{pred_dir}_classifiers/{repeat}_{random_fold}_model.joblib'))

            averaging_clf = PrefitAveragingClassifier(model_list)

            y_prob = averaging_clf.predict_proba(X_re)[:, 1]

            # Permutation importance
            per_result = permutation_importance(averaging_clf, X_re, y_re, n_repeats=50, scoring='roc_auc', n_jobs=-1)

            performance.inner_loop(y_test=y_re,
                                   y_prob=averaging_clf.predict_proba(X_re)[:, 1],
                                   importances=per_result.importances_mean)

            # Update the progress
            print(f'Repeat {repeat} fold {i_fold} done')

        # Average the iteration scores and add to the list of outer cv results
        performance.end_inner()

    importance_names = load(f'{pred_root}{pred_target_dirs[0]}/rf_{pred_target_dirs[0]}_classifiers/importances_column_names.joblib')

    performance.create_final(target_dir)
    performance.create_importances(target_dir, importance_names)
