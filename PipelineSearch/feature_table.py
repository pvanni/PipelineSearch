import pandas as pd
import numpy as np
import os
from PipelineSearch.util import ShellCommand


def filtering_table(table_dir, database, temp_id=None):
    """Take the feature table and apply taxonomic filtering and chimera removal. Replace the original repseq and table
    in the feature_table/ directory with the filtered ones"""

    os.mkdir(f'{table_dir}/{database}')

    # Chimera filtering here for much faster preprocessing
    ShellCommand('qiime vsearch uchime-denovo '
                 f'--i-table {table_dir}/feature_table.qza '
                 f'--i-sequences {table_dir}/representative_sequences.qza '
                 f'--output-dir temp/{temp_id}_chimera_removal')


    # Remove chimeric and borderline chimeric from table and rep_seqs
    ShellCommand('qiime feature-table filter-features '
                 f'--i-table {table_dir}/feature_table.qza '
                 f'--m-metadata-file temp/{temp_id}_chimera_removal/nonchimeras.qza '
                 f'--o-filtered-table {table_dir}/{database}/feature_table.qza')


    ShellCommand('qiime feature-table filter-seqs '
                 f'--i-data {table_dir}/representative_sequences.qza '
                 f'--m-metadata-file temp/{temp_id}_chimera_removal/nonchimeras.qza '
                 f'--o-filtered-data {table_dir}/{database}/representative_sequences.qza')


    ShellCommand('qiime feature-table filter-samples '
                 f'--i-table {table_dir}/{database}/feature_table.qza '
                 '--p-min-frequency 1000 '
                 f'--o-filtered-table {table_dir}/{database}/feature_table.qza')

    ShellCommand('qiime feature-table filter-features '
                 f'--i-table {table_dir}/{database}/feature_table.qza '
                 '--p-min-samples 2 '
                 '--p-min-frequency 10 '
                 f'--o-filtered-table {table_dir}/{database}/feature_table.qza')


    ShellCommand('qiime feature-classifier classify-sklearn '
                 f'--i-classifier taxonomic_classifier/{database}_classifier.qza '
                 f'--i-reads {table_dir}/{database}/representative_sequences.qza '
                 f'--o-classification {table_dir}/{database}/taxonomy.qza')


    # Filter out mitochondria and chloroplast sequences
    ShellCommand('qiime taxa filter-table '
                 f'--i-table {table_dir}/{database}/feature_table.qza '
                 f'--i-taxonomy {table_dir}/{database}/taxonomy.qza '
                 f'--p-exclude mitochondria,chloroplast '
                 f'--o-filtered-table {table_dir}/{database}/feature_table.qza ')

    ShellCommand('qiime taxa filter-table '
                 f'--i-table {table_dir}/{database}/feature_table.qza '
                 f'--i-taxonomy {table_dir}/{database}/taxonomy.qza '
                 f'--p-include Bacteria '
                 f'--o-filtered-table {table_dir}/{database}/feature_table.qza ')

    # Representative sequences aswell

    ShellCommand('qiime taxa filter-seqs '
                 f'--i-sequences {table_dir}/{database}/representative_sequences.qza '
                 f'--i-taxonomy {table_dir}/{database}/taxonomy.qza '
                 f'--p-exclude mitochondria,chloroplast '
                 f'--o-filtered-sequences {table_dir}/{database}/representative_sequences.qza ')

    ShellCommand('qiime taxa filter-seqs '
                 f'--i-sequences {table_dir}/{database}/representative_sequences.qza '
                 f'--i-taxonomy {table_dir}/{database}/taxonomy.qza '
                 f'--p-include Bacteria '
                 f'--o-filtered-sequences {table_dir}/{database}/representative_sequences.qza ')


    return


def picrust2_table(table_dir, threads=1, temp_id=None):


    os.mkdir(f'{table_dir}/pathways')

    # Save all the file names
    table, repseq = f'{table_dir}/feature_table.qza', f'{table_dir}/representative_sequences.qza'

    ShellCommand('qiime picrust2 full-pipeline '
                 f'--i-table {table} '
                 f'--i-seq {repseq} '
                 f'--o-ko-metagenome temp/{temp_id}_ko_meta.qza '
                 f'--o-ec-metagenome temp/{temp_id}_ec_meta.qza '
                 f'--o-pathway-abundance {table_dir}/pathways/feature_table.qza '
                 f'--p-threads {threads} '
                 '--p-hsp-method mp '
                 '--p-max-nsti 2')

    return


def collapse_table(table_dir, taxonomy_dir=None, level=None, temp_id=None):

    if level:
        os.mkdir(f'{table_dir}/level{level}')

        ShellCommand('qiime taxa collapse '
                     f'--i-table {table_dir}/feature_table.qza '
                     f'--i-taxonomy {taxonomy_dir}/taxonomy.qza '
                     f'--p-level {level} '
                     f'--o-collapsed-table {table_dir}/level{level}/feature_table.qza')
    else:
        os.mkdir(f'{table_dir}/no_collapsing')
        ShellCommand(f'cp {table_dir}/feature_table.qza {table_dir}/no_collapsing/feature_table.qza')

    return


def output_table(table_dir, relative='non_relative', temp_id=None):

    os.mkdir(f'{table_dir}/{relative}')

    if relative == 'relative':
        ShellCommand('qiime feature-table relative-frequency '
                     f'--i-table {table_dir}/feature_table.qza '
                     f'--o-relative-frequency-table {table_dir}/{relative}/feature_table.qza')
    else:
        ShellCommand(f'cp {table_dir}/feature_table.qza {table_dir}/{relative}/feature_table.qza')

    ShellCommand('qiime tools export '
                 f'--input-path {table_dir}/{relative}/feature_table.qza '
                 f'--output-path temp/{temp_id}_exported')

    ShellCommand('biom convert '
                 f'-i temp/{temp_id}_exported/feature-table.biom '
                 f'-o {table_dir}/{relative}/feature_table.tsv '
                 '--to-tsv')

    return


def convert_to_ml(table_dir, target_col, pos, neg, sample_column='#SampleID', prevalence_threshold=0, temp_id=None):
    """Convert both relative abundance picrust2 and taxa feature tables"""

    def feature_check(data, feature, threshold):
        """Checks if the feature is in the data and if it passes the threshold
        returns True if the check passes"""
        if feature in data.columns.values:
            # Number of nonzero samples
            non_zero = len(np.nonzero(data[feature].values)[0])
            if non_zero / data.shape[0] >= threshold:
                return True
            else:
                return False
        else:
            return False

    df = pd.read_csv(f'{table_dir}/feature_table.tsv', sep='\t', header=1)
    meta = pd.read_csv(f'start_files/metadata.tsv', sep='\t')

    # Make it so the header is feature names and rows are samples in df
    df = df.T
    df.columns = df.iloc[0, :].values
    df = df.drop(['#OTU ID'], axis=0)

    # do prevalence filtering if its selected
    if prevalence_threshold > 0:
        feature_survived = []
        for feature in df.columns.values:
            feature_survived.append(feature_check(df, feature, prevalence_threshold))
        df = df.loc[:, feature_survived]

        feature_list = []
        for feature, did_it_pass in zip(df.columns.values, feature_survived):
            if feature == 'Classes':
                continue
            if did_it_pass:
                feature_list.append(feature)

        df = df[feature_list]

        print(f'Threshold {prevalence_threshold} retained {len(feature_list)} features')

    # Convert the index col names to strings so they match with the qiime2 export
    index_names = []
    for i in meta[sample_column].values:
        index_names.append(str(i))

    meta.index = index_names
    meta = meta.loc[df.index.values]

    classes = []
    boolean_list = []
    # Take only the target samples
    for i in meta[target_col].values:
        if str(i) in [str(pos), str(neg)]:
            boolean_list.append(True)
        else:
            boolean_list.append(False)

    meta = meta.loc[boolean_list]
    df = df.loc[boolean_list]

    for i in meta[target_col].values:
        if str(i) == str(pos):
            classes.append(1)
        else:
            classes.append(0)

    df['Classes'] = classes
    print(np.unique(classes, return_counts=True))

    df.to_csv(f'{table_dir}/prev_{prevalence_threshold}_ML_table.tsv', index=True, sep='\t')
    return
