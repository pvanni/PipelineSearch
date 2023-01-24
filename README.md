Quick overview of scripts and their usage:

"analyze.py" was used ran to produce all the necessary Qiime2 preprocessing and filtering step up to ML analyses.
This script uses the PipelineSearch package. Need to be set up and ran separately for each cohort.

"ML_uniform_checker.py" was used to remove unique features from cohorts, as the cross-study analyses cannot be done if the feature tables are not the same size in all cohorts.

"nested_cv_pipeline.py" stand-alone script used for each ML feature table produced from analyze.py. 
Requires an feature table where rows are samples and columns are features. Last column needs to be the target variable, which in our case was a binary variable about the samples delivery mode. This script uses joblib to dump each classifier for feature importance calculations or cross-study analyses.

"cross_study
