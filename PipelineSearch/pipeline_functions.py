from PipelineSearch.util import ShellCommand
from PipelineSearch.util import return_negative_rep_seqs
import PipelineSearch.preprocessing_functions as preprocessing
import PipelineSearch.taxonomy as taxonomy
import PipelineSearch.taxonomy as taxonomy
import PipelineSearch.feature_table as table
from PipelineSearch.util import tempfile_cleaner
import multiprocessing as mp
import os
import pandas as pd
import numpy as np

def closed_ref_unit(f_primer, r_primer, paired_sequences, manifest_type, threshold=0.8):
    # Import data
    preprocessing.import_function(manifest_type=manifest_type,
                                  paired=paired_sequences,
                                  error_rate=0,
                                  sample_column='#SampleID')



    # Remove primers
    chosen_trunc_len = preprocessing.primer_removal(forward_list=f_primer,
                                                    reverse_list=r_primer,
                                                    threshold=threshold,
                                                    no_trunc=True,
                                                    paired=paired_sequences)

    # Filter for clustering
    preprocessing.filter_preclustering(phred_threshold=1,
                                       lower_limit=100,
                                       paired=paired_sequences)

    database = 'greengenes'

    preprocessing.closed_clustering(database=database,
                                    ref_seqs=f"taxonomic_database/{database}.fasta",
                                    perc_identity=0.99)

    ShellCommand('qiime feature-table summarize '
                 '--i-table closed_greengenes/feature_table.qza '
                 '--o-visualization temp/closed_ref_table.qzv')

    return


def preprocessing_pipeline(f_primer, r_primer, paired_sequences, trim_taxonomy, tax_databases, no_trunc, tax_trunc_len, manifest_type, threshold=0.8, primer_error_rate=0.1):
    """Does all the commands that need to be executed only once per dataset"""

    # Import data
    preprocessing.import_function(manifest_type=manifest_type,
                                  paired=paired_sequences,
                                  error_rate=0,
                                  sample_column='#SampleID')

    # Remove primers
    chosen_trunc_len = preprocessing.primer_removal(forward_list=f_primer,
                                                    reverse_list=r_primer,
                                                    threshold=threshold,
                                                    no_trunc=no_trunc,
                                                    paired=paired_sequences,
                                                    error_rate=primer_error_rate)

    """
    # Filter for clustering
    preprocessing.filter_preclustering(phred_threshold=20,
                                       paired=paired_sequences)
    """
    dada_used_len = preprocessing.dada2(trunc_len=chosen_trunc_len,
                                       trim_left=0,
                                       paired=paired_sequences,
                                       threshold=threshold)

    # Used in Houston data. Sequences were flipped
    #return_negative_rep_seqs('dada2/representative_sequences.qza')

    # For denovo clustering. Not used in the article
    #preprocessing.denovo_clustering(perc_identity=0.99)

    for database in tax_databases:
        """ Not used in the article
        preprocessing.closed_clustering(database=database,
                                                       ref_seqs=f"taxonomic_database/{database}.fasta",
                                                       perc_identity=0.99)
        """

        #return_negative_rep_seqs(f'closed_{database}/representative_sequences.qza')

        taxonomy.train_taxonomic_classifier(f_primer=f_primer[0],
                                                 r_primer=r_primer[0],
                                                 trunc_len=tax_trunc_len,
                                                 trim_left=0,
                                                trim_taxonomy=trim_taxonomy,
                                                 rep_file=f"taxonomic_database/{database}.fasta",
                                                 tax_file=f"taxonomic_database/{database}.txt",
                                                 database_name=database)

    return


def return_function(name):
    """Function that connects function names to actual functions"""
    function_dict = {
        'dada2': preprocessing.dada2,
        'denovo': preprocessing.denovo_clustering,
        'closed_silva': preprocessing.closed_clustering,
        'closed_greengenes': preprocessing.closed_clustering,
        'filter': table.filtering_table,
        'genera': table.collapse_table,
        'pathways': table.picrust2_table,
        'no_collapsing': table.collapse_table,
        'relative': table.output_table,
        'non_relative': table.output_table,
        'prevalence': table.convert_to_ml
    }


    return function_dict[name]


def return_path(path_list):

    path = path_list[0]
    for word in path_list[1:]:
        if len(word) > 1:
            path += f'/{word}'

    return path




def parallelization_controller(function_dict, param_dict, num_workers, step_list):

    def run_function(func, kwargs):
        print(f'Starting {func} with arguments {kwargs}')
        func(**kwargs)
        return

    def split_step_list(num_workers, step_list):
        job_lists = []

        while len(step_list) > 0:
            job = []
            if len(step_list) > num_workers:
                for i in range(num_workers):
                    job.append(step_list.pop(0))
            else:
                job = step_list
                step_list = []
            job_lists.append(job)

        return job_lists



    job_lists = split_step_list(num_workers, step_list)
    process_list = []
    for jobs in job_lists:
        for step in jobs:
            proc = mp.Process(target=run_function, args=(function_dict[f'step{step}'], param_dict[f'step{step}']))
            proc.start()
            process_list.append(proc)
        for process_item in process_list:
            process_item.join()
    return


def create_function_grid(clustering_method_list,
                         tax_database_list,
                         feature_table_type_list,
                         relative_list,
                         prevalence_threshold_list,
                         ML_target_col,
                         ML_pos_class,
                         ML_neg_class,
                         sample_column='#SampleID',
                         num_workers=1):
    """
    Example choices:
    clustering_method_list=     ['dada2', 'denovo', 'closed_silva', 'closed_greengenes'],
    tax_database_list=          ['tax_silva', 'tax_greengenes'],
    feature_table_type_list=    ['genera', 'pathways', 'no_collapsing'],
    relative_list=              ['relative', 'non_relative'],
    prevalence_threshold=       [0.01, 0.1, 0.3]
    """
    function_dict = {}
    param_dict = {}

    step_counter = 0
    for clustering_method in clustering_method_list:
        current_path = [clustering_method, '', '', '']

        # Add table filtering functions and their parameters
        for database in tax_database_list:
            function_dict[f'step{step_counter}'] = return_function('filter')
            param_dict[f'step{step_counter}'] = {'table_dir': return_path(current_path),
                                                 'database': database,
                                                 'temp_id': step_counter}
            current_path[1] = database

            # Record the path to taxonomy file
            taxonomy_dir = current_path

            # Add a step
            step_counter += 1

            # Which type of feature table is it going to be
            for table_type in feature_table_type_list:
                function_dict[f'step{step_counter}'] = return_function(table_type)

                if table_type == 'genera':
                    param_dict[f'step{step_counter}'] = {'table_dir': return_path(current_path),
                                                         'level': 6,
                                                         'taxonomy_dir': return_path(taxonomy_dir),
                                                         'temp_id': step_counter}
                    current_path[2] = 'level6'
                else:
                    param_dict[f'step{step_counter}'] = {'table_dir': return_path(current_path),
                                                         'temp_id': step_counter}
                    current_path[2] = table_type

                # Add a step
                step_counter += 1

                # Relative abundance or not
                for relative in relative_list:
                    function_dict[f'step{step_counter}'] = return_function(relative)

                    param_dict[f'step{step_counter}'] = {'table_dir': return_path(current_path),
                                                         'relative': relative,
                                                         'temp_id': step_counter}
                    current_path[3] = relative

                    # Add a step
                    step_counter += 1

                    # Convert to machine learning feature table
                    for prevalence_threshold in prevalence_threshold_list:

                        function_dict[f'step{step_counter}'] = return_function('prevalence')

                        param_dict[f'step{step_counter}'] = {'table_dir': return_path(current_path),
                                                             'target_col': ML_target_col,
                                                             'pos': ML_pos_class,
                                                             'neg': ML_neg_class,
                                                             'sample_column': sample_column,
                                                             'prevalence_threshold': prevalence_threshold,
                                                             'temp_id': step_counter
                                                             }

                        step_counter += 1

                    current_path[3] = ''
                current_path[2] = ''
            current_path[1] = ''


    print(f'Number of workers {num_workers}')

    exe_order = ['filtering_table', 'collapse_table', 'picrust2_table', 'output_table', 'convert_to_ml']
    for func in exe_order:
        step_list = []
        for i in range(step_counter):
            if func in str(function_dict[f'step{i}']).split(' ')[1]:
                step_list.append(i)

        # Use process

        parallelization_controller(function_dict, param_dict, num_workers, step_list)

    tempfile_cleaner('temp')


def gather_table_paths(dir_path):

    dir_list = os.listdir(dir_path)

    feature_table_list = []

    for dir in dir_list:
        if os.path.isdir(f'{dir_path}/{dir}'):
            next_path = f'{dir_path}/{dir}'
            feature_table_list += gather_table_paths(next_path)
        if '_ML_table.tsv' in dir:
            feature_table_list += [f'{dir_path}/{dir}']

    return feature_table_list


def combine_ML_tables(table_path_list):

    df_list = []

    # Add a tag to each feature so they can be seperated from other feature tables
    for path in table_path_list:
        df = pd.read_csv(path, index_col=0, sep='\t')

        tag = path[len(os.getcwd()):]

        new_col = []
        for col in df.columns.values:
            if col == 'Classes':
                new_col.append(col)
            else:
                new_col.append(tag + '/' + col)

        df.columns = new_col
        df_list.append(df)

    # Record how many samples are in each df
    df_n = []
    for df in df_list:
        df_n.append(df.shape[0])

    # Remove df's that dont have the same amount of samples as the majority
    unique_n, counts = np.unique(df_n, return_counts=True)

    print(df_n)
    print(counts)
    majority_n = unique_n[np.argmax(counts)]

    # Record the feature tables that survived
    survived_paths = []
    # Take only the df's that have the majority amount of samples
    same_n_df_list = []
    for df, n, path in zip(df_list, df_n, table_path_list):
        if n == majority_n:
            survived_paths.append(path)
            same_n_df_list.append(df)

    # Now combine all the dataframes into the same .tsv file with Classes feature in the end
    final_df = same_n_df_list.pop(0)
    final_class = final_df['Classes'].values
    final_df.drop(columns=['Classes'], inplace=True)
    for df in same_n_df_list:
        final_df = final_df.join(df.drop(columns='Classes'))

    final_df['Classes'] = final_class

    # Write the names of the feature tables that survived into a file
    with open(f'feature_tables_survived.txt', 'w+') as writefile:

        for path in survived_paths:
            writefile.write(f'{path}\n')

    final_df.to_csv('all_ML_feature_tables.tsv', sep='\t', index=True)
    return
