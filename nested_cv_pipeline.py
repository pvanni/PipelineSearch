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
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


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


class curve_and_metrics:
    """Class for gathering metrics and curve plotting"""

    def __init__(self, loops):

        # Variables for precision-recall curves
        self.pooled_y = []
        self.pooled_prob = []

        # Dummy values for baseline
        self.y_dummy_prob = []

        # Variables for importance gathering
        self.outer_importances = []
        self.inner_importances = []

        # Mean false positive rate for roc curves
        self.mean_fpr = np.linspace(0, 1, 100)
        self.inner_tprs = []
        self.inner_aucs = []
        self.inner_dummy_aucs = []
        self.inner_dummy_tprs = []

        # Outer
        self.outer_tprs = []
        self.outer_aucs = []
        self.outer_dummy_aucs = []
        self.outer_dummy_tprs = []

        self.p_list = []

    def inner_loop(self, y_test, y_prob, y_dummy_prob, importances):
        """Add values from each nested cv outer fold (meaning innner)
        outer means the iteration averaged values of a nested cv"""

        self.inner_importances.append(importances)
        self.pooled_y.append(y_test)
        self.pooled_prob.append(y_prob)

        self.y_dummy_prob.append(y_dummy_prob)

        # Generate roc curves and auc values
        real_tpr, real_auc = self.tpr_and_auc(y_test, y_prob)
        dummy_tpr, dummy_auc = self.tpr_and_auc(y_test, y_dummy_prob)

        # Add real values
        self.inner_tprs.append(real_tpr)
        self.inner_aucs.append(real_auc)

        # Add dummy values
        self.inner_dummy_tprs.append(dummy_tpr)
        self.inner_dummy_aucs.append(dummy_auc)


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

        mean_tpr_dummy = np.mean(self.inner_dummy_tprs, axis=0)
        mean_tpr_dummy[-1] = 1.0

        # Add dummy values
        self.outer_dummy_tprs.append(mean_tpr_dummy)
        self.outer_dummy_aucs.append(np.mean(self.inner_dummy_aucs))

        # Reset the inner lists
        self.inner_importances = []
        self.inner_tprs = []
        self.inner_aucs = []
        self.inner_dummy_aucs = []
        self.inner_dummy_tprs = []

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

            mean_tpr_dummy = np.mean(self.outer_dummy_tprs, axis=0)
            mean_auc_dummy = auc(self.mean_fpr, mean_tpr_dummy)
            std_auc_dummy = np.std(self.outer_dummy_aucs)

            writefile.write(f'Dummy classifier ROC_AUC: {mean_auc_dummy}\n'
                            f'Dummy classifier ROC_AUC_STD: {std_auc_dummy}\n')

            std_tpr_dummy = np.std(self.outer_dummy_tprs, axis=0)
            tprs_upper_dummy = np.minimum(mean_tpr_dummy + std_tpr_dummy, 1)
            tprs_lower_dummy = np.maximum(mean_tpr_dummy - std_tpr_dummy, 0)

            # Add them to roc_curve_df for saving
            roc_curve_df['True positive rate (Dummy)'] = mean_tpr_dummy
            roc_curve_df['Lower confidence interval (Dummy)'] = tprs_lower_dummy
            roc_curve_df['Upper confidence interval (Dummy)'] = tprs_upper_dummy

            roc_curve_df.to_csv(f'{file_name}_roc_curve.tsv', index=False, sep='\t')


            # Make the precision-recall curve
            y_real = np.concatenate(self.pooled_y)
            y_dummy_prob = np.concatenate(self.y_dummy_prob)
            y_proba = np.concatenate(self.pooled_prob)

            # Real classifier
            precision, recall, _ = precision_recall_curve(y_real, y_proba)
            pr_curve_df = pd.DataFrame(data={'precision': precision,
                                             'recall': recall})
            pr_auc = average_precision_score(y_real, y_proba)

            pr_curve_df.to_csv(f'{file_name}_pr_curve.tsv', index=False, sep='\t')

            # Dummy classifier
            precision_dummy, recall_dummy, _ = precision_recall_curve(y_real, y_dummy_prob)
            pr_curve_df_dummy = pd.DataFrame(data={'precision': precision_dummy,
                                                   'recall': recall_dummy})
            pr_auc_dummy = average_precision_score(y_real, y_dummy_prob)

            pr_curve_df_dummy.to_csv(f'{file_name}_dummy_pr_curve.tsv', index=False, sep='\t')

            writefile.write(f'Real classifier PR_AUC: {pr_auc}\n'
                            f'Dummy classifier PR_AUC: {pr_auc_dummy}\n')

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



class nested_cv_full_pipe:
    """Nested cross-validation model tuning with permutation tests versus random"""


    def __init__(self, input_df,
                 alg_name,
                 param_grid,
                 rand_search,
                 repeats=10,
                 splits=None,
                 file_name='testi',
                 inverse_scoring=False,
                 permutation_testing=False,
                 multicollinear=None):

        if splits is None:
            splits = [10, 10]

        if inverse_scoring:
            print('Inversing ROC_AUC scoring')
            scorer = make_scorer(roc_auc_score, needs_threshold=True, greater_is_better=False)
        else:
            scorer = 'roc_auc'

        df = pd.read_csv(f"{input_df}", index_col=0, sep='\t')

        self.core_method(df=df,
                         param_grid=param_grid,
                         alg_name=alg_name,
                         file_name=file_name,
                         splits=splits,
                         repeats=repeats,
                         scoring=scorer,
                         rand_search=rand_search,
                         permutation_testing=permutation_testing,
                         multicollinear=multicollinear)

    def return_model(self, alg_name, param_grid, cv, scoring, col_names, rand_search=False, rng=None):
        """Return the gridsearchcv object containing the right estimator and parameter grid"""
        classifier = 0

        if alg_name == 'extratreesclassifier':
            classifier = ExtraTreesClassifier(n_estimators=250,
                                              n_jobs=-1,
                                              random_state=rng)
        if alg_name == 'adaboostclassifier':
            classifier = AdaBoostClassifier(random_state=rng)

        if alg_name == 'logisticregression':
            classifier = LogisticRegression(random_state=rng)

        if alg_name == 'svc':
            classifier = SVC(probability=True)

        if alg_name == 'randomforestclassifier':
            classifier = RandomForestClassifier(n_estimators=250,
                                                n_jobs=-1,
                                                random_state=rng)
        if alg_name == 'mlpclassifier':
            classifier = MLPClassifier(random_state=rng)

        if alg_name == 'lgbmclassifier':
            classifier = LGBMClassifier(n_jobs=-1, random_state=rng)

        pipe = Pipeline([('standardscaler', StandardScaler()),
                         ('pipelineselector', PipelineSelector(column_names=col_names)),
                         ('vifselector', VifSelector(random_state=rng)),
                         ('resampler', SMOTE(random_state=rng)),
                         (f'{alg_name}', classifier)])

        if rand_search:
            gs = RandomizedSearchCV(estimator=pipe,
                                    param_distributions=param_grid,
                                    scoring=scoring,
                                    n_iter=rand_search,
                                    cv=cv,
                                    n_jobs=-1,
                                    iid=False,
                                    random_state=rng)
        else:
            gs = GridSearchCV(estimator=pipe,
                              param_grid=param_grid,
                              scoring=scoring,
                              cv=cv,
                              n_jobs=-1,
                              iid=False)

        return gs

    def core_method(self, df, alg_name, param_grid, file_name, rand_search, splits=None,
                    repeats=10, scoring='roc_auc', permutation_testing=False,
                    rng=None,
                    split_rng=None,
                    multicollinear=None):
        """Run the outer fold cross-validation. Create curves, averaged feature importances and other metrics"""

        # Determine if 10 fold CV is too high

        # So that the argument is not mutable
        if splits is None:
            splits = [10, 10]

        rng = np.random.RandomState(11102021)
        split_rng = np.random.RandomState(16102021)

        min_splits = np.min(np.unique(df['Classes'].values, return_counts=True)[1])
        print(f'There are {min_splits} minority samples')

        performance = curve_and_metrics(loops=repeats)

        # Create a folder that contains all the outerfold models pickled
        os.mkdir(f'{file_name}_classifiers')

        for rep in range(repeats):

            df_shuffled = shuffle(df, random_state=split_rng)

            # Samples with features
            X = df_shuffled.iloc[:, :len(df.columns) - 1].values
            # Samples with class labels
            y = df_shuffled.iloc[:, len(df.columns) - 1].values

            # Create the outer folds
            skf = StratifiedKFold(n_splits=splits[1], shuffle=True, random_state=split_rng)

            pickle_counter = 0

            for train, test in skf.split(X, y):

                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]

                # Return the gridsearch object with the right estimator and settings
                model = self.return_model(alg_name, param_grid,
                                          StratifiedKFold(n_splits=splits[0], shuffle=True, random_state=rng),
                                          scoring,
                                          col_names=df.columns.values[:-1],
                                          rand_search=rand_search,
                                          rng=rng)

                # Fit the model and test on outer fold
                model.fit(X_train, y_train)

                # Baseline classifier
                dummy = DummyClassifier(strategy='stratified', random_state=rng)
                dummy.fit(X_train, y_train)

                # Save the classifier for possible later use. Also pickle the test set for permutation testing
                dump(model, f'{file_name}_classifiers/{rep}_{pickle_counter}_model.joblib')
                dump(X_test, f'{file_name}_classifiers/{rep}_{pickle_counter}_Xtest.joblib')
                dump(y_test, f'{file_name}_classifiers/{rep}_{pickle_counter}_Ytest.joblib')
                pickle_counter += 1

                # Record performances
                importances = [0 for i in range(len(df.columns.values[:-1]))]
                if alg_name in ['lgbmclassifier', 'randomforestclassifier', 'extratreesclassifier', 'adaboostclassifier']:
                    alg_importances = model.best_estimator_[alg_name].feature_importances_
                else:
                    #alg_importances = model.best_estimator_[alg_name].coef_
                    alg_importances = importances

                if not param_grid['pipelineselector'] == [None]:
                    # Connect the model importances to the whole set of features
                    best_feature_indeces = model.best_estimator_['pipelineselector'].feature_indeces

                    for feat_i, alg_importance_value in zip(best_feature_indeces, alg_importances):
                        importances[feat_i] = alg_importance_value
                elif multicollinear:
                    # Connect the model importances to the whole set of features
                    best_feature_indeces = model.best_estimator_['vifselector'].feature_indeces

                    for feat_i, alg_importance_value in zip(best_feature_indeces, alg_importances):
                        importances[feat_i] = alg_importance_value
                else:
                    importances = alg_importances

                if scoring == 'roc_auc':
                    performance.inner_loop(y_test=y_test,
                                           y_prob=model.predict_proba(X_test)[:, 1],
                                           y_dummy_prob=dummy.predict_proba(X_test)[:, 1],
                                           importances=importances)
                else:
                    performance.inner_loop(y_test=y_test,
                                           y_prob=model.predict_proba(X_test)[:, 0],
                                           y_dummy_prob=dummy.predict_proba(X_test)[:, 0],
                                           importances=importances)

            # Average the iteration scores and add to the list of outer cv results
            performance.end_inner()

            if not permutation_testing:
                print('Skipping permutation testing')
                continue

            per_cv = StratifiedKFold(n_splits=splits[0], shuffle=True, random_state=rng)

            # Do permutation testing
            score, permutation_scores, pvalue = permutation_test_score(self.return_model(alg_name,
                                                                                         param_grid,
                                                                                         cv=per_cv,
                                                                                         scoring=scoring,
                                                                                         col_names=df.columns.values[:-1],
                                                                                         rand_search=rand_search,
                                                                                         rng=rng),
                                                                       X=X,
                                                                       y=y,
                                                                       scoring=scoring,
                                                                       cv=StratifiedKFold(n_splits=splits[1],
                                                                                          shuffle=True),
                                                                       n_permutations=30,
                                                                       n_jobs=-1,
                                                                       random_state=rng)

            # Calculate the p-value of the permutation test using the earlier cv result

            if scoring == 'roc_auc':
                performance.permutation_p(permutation_scores)
            else:
                permutation_scores = 1 + permutation_scores
                performance.permutation_p(permutation_scores)

        performance.create_final(file_name)
        performance.create_importances(file_name, df.columns.values[:-1])

        #Pickle also the importance names
        dump(df.columns.values[:-1], f'{file_name}_classifiers/importances_column_names.joblib')


# Create param_grid of MLP architectures
hidden_layer_sizes_list = []
for n_layer in [1, 2, 3]:
    for n_neurons in [10, 30, 50, 100]:
        param_tuple = 0
        if n_layer == 1:
            param_tuple = (n_neurons,)

        if n_layer == 2:
            param_tuple = (n_neurons, int(n_neurons/2))

        if n_layer == 3:
            param_tuple = (n_neurons, int(n_neurons/2), int(n_neurons/4))
        hidden_layer_sizes_list.append(param_tuple)

param_list = [
              {'mlpclassifier__hidden_layer_sizes': hidden_layer_sizes_list,
               'mlpclassifier__max_iter': [100, 200, 300, 500, 700],
               'mlpclassifier__solver': ['adam'],
               'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
               'mlpclassifier__learning_rate_init': [0.001, 0.01, 0.05],
               'mlpclassifier__momentum': [0.6, 0.7, 0.8, 0.9],
               'pipelineselector': [None],
               'vifselector': [None]},
              {'lgbmclassifier__num_leaves': [15, 31, 80],
               'lgbmclassifier__max_depth': [-1, 5, 10, 20],
               'lgbmclassifier__n_estimators': [50, 100, 150, 200],
               'lgbmclassifier__reg_alpha': [0, 0.01, 0.03, 0.05],
               'lgbmclassifier__learning_rate': [0.05, 0.1, 0.15, 0.2],
               'lgbmclassifier__is_unbalance': [True],
               'resampler': [None],
               'pipelineselector': [None],
               'vifselector': [None]},
              {'randomforestclassifier__max_depth': [None, 5, 10],
               'randomforestclassifier__max_features': ['auto', 15, None],
               'randomforestclassifier__class_weight': ['balanced', 'balanced_subsample'],
               'randomforestclassifier__bootstrap': [True, False],
               'resampler': [None],
               'pipelineselector': [None],
               'vifselector': [None]},
              {'extratreesclassifier__max_depth': [None, 5, 10],
               'extratreesclassifier__max_features': ['auto', 15, None],
               'extratreesclassifier__class_weight': ['balanced', 'balanced_subsample'],
               'resampler': [None],
               'pipelineselector': [None],
               'vifselector': [None]}]

alg_list = ['mlpclassifier', 'lgbmclassifier', 'randomforestclassifier', 'extratreesclassifier']

# Random state

for group_name in [f'{sys.argv[2]}']:

    tag = f'{group_name}'

    file_list = [f'mlp_{tag}', f'lgbm_{tag}', f'rf_{tag}', f'extra_{tag}']

    for alg_name, file_name, param_grid in zip(alg_list, file_list, param_list):
        
        nested_cv_full_pipe(input_df=sys.argv[1],
                            alg_name=alg_name,
                            repeats=40,
                            param_grid=param_grid,
                            splits=[10, 10],
                            file_name=file_name,
                            inverse_scoring=False,
                            rand_search=40,
                            permutation_testing=False,
                            multicollinear=None)




