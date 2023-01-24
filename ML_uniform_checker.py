import os
import pandas as pd
import numpy as np


def make_uniform_df(df_list_input):

    all_col_names = []

    for df in df_list_input:
        all_col_names += list(df.columns.values)[:-1]

    all_col_names = np.unique(all_col_names, return_counts=True)

    found_in_all = []
    for col, n_count in zip(all_col_names[0], all_col_names[1]):
        if n_count == len(df_list_input):
            found_in_all.append(col)

    found_in_all.append('Classes')

    new_df_list = []
    for df2 in df_list_input:
        new_df_list.append(df2[found_in_all])

    return new_df_list


path_taxa = '/dada2/silva/level6/relative/prev_0_ML_table.tsv'
path_pathways = '/dada2/silva/pathways/relative/prev_0_ML_table.tsv'
for month in ['_month', '6month', 'year']:

    # Gather all the target datasets folders
    dir_list = os.listdir()
    target_dirs = []

    for dir_name in dir_list:
        if month in dir_name:
            target_dirs.append(dir_name)

    # Create directories for taxa and pathways
    for df_path, description in zip([path_taxa, path_pathways], ['taxa', 'pathways']):

        # create folder for each sample time
        os.mkdir(f'{month}_{description}_baseline')

        # Collect all the target dataframes into a list
        df_list = []
        for target_dir in target_dirs:
            df = pd.read_csv(f'{target_dir}{df_path}', index_col=0, sep='\t')
            df_list.append(df)

        uniform_df_list = make_uniform_df(df_list)

        for df, file_name in zip(uniform_df_list, target_dirs):
            # create folder for each sample time
            os.mkdir(f'{month}_{description}_baseline/{file_name}')
            df.to_csv(f'{month}_{description}_baseline/{file_name}/{file_name}_ML.tsv', index=True, sep='\t')
