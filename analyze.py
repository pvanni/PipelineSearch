import PipelineSearch as pipe
import sys
import os

# Used for easy restart for debugging
if 'restart' in sys.argv:
    pipe.util.restart(os.getcwd())
    print('Project directory cleaned and reset')
    exit(0)

f_primer=["XXXXXXXXXXXXXXXXXXXXXX"] # Specify your forward primer here
r_primer=["XXXXXXXXXXXXXXXXXXXXXXXXX"] # Specify your reverse primer here, if any

# Example
pipe.pipeline_functions.preprocessing_pipeline(f_primer=f_primer,
                                               r_primer=[None],
                                               paired_sequences=False,
                                               manifest_type=True,
                                               threshold=0.8,
                                               primer_error_rate=0.1,
                                               trim_taxonomy=True,
                                               no_trunc=True,
                                               tax_trunc_len=0,
                                               tax_databases=['silva', 'greengenes'])


pipe.pipeline_functions.create_function_grid(clustering_method_list=['dada2'],
                                             tax_database_list=['silva', 'greengenes'],
                                             feature_table_type_list=['pathways', 'genera', 'no_collapsing'],
                                             relative_list=['relative', 'non_relative'],
                                             prevalence_threshold_list=[0, 0.05, 0.1, 0.2],
                                             ML_target_col='Delivery_mode',
                                             ML_pos_class='C-section',
                                             ML_neg_class='Vaginal',
                                             sample_column='#SampleID',
                                             num_workers=4)

results = pipe.pipeline_functions.gather_table_paths(dir_path=os.getcwd())

pipe.pipeline_functions.combine_ML_tables(results)
