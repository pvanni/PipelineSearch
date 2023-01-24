import os
from PipelineSearch.util import ShellCommand
from PipelineSearch.util import para_writer


def train_taxonomic_classifier(f_primer, r_primer, trunc_len, trim_left, trim_taxonomy, rep_file, tax_file, database_name):
    """Take the chosen settings from denoising() and train the taxonomic classifier"""
    paragraph_list = []
    project_path = os.getcwd()

    if not os.path.isdir(f'{project_path}/taxonomic_classifier'):
        os.mkdir(f'{project_path}/taxonomic_classifier')

    if os.path.isfile(f'{project_path}/taxonomic_classifier/{database_name}_classifier.qza'):
        print(f'{database_name}_classifier.qza already exists. Exiting the function')
        return

    ShellCommand("qiime tools import "
                 "--type 'FeatureData[Sequence]' "
                 f"--input-path {rep_file} "
                 f"--output-path {project_path}/temp/rep_file.qza")

    ShellCommand("qiime tools import "
                 "--type 'FeatureData[Taxonomy]' "
                 "--input-format HeaderlessTSVTaxonomyFormat "
                 f"--input-path {tax_file} "
                 f"--output-path {project_path}/temp/tax_file.qza")
    if trim_taxonomy:
        ShellCommand("qiime feature-classifier extract-reads "
                     f"--i-sequences {project_path}/temp/rep_file.qza "
                     f"--p-f-primer {f_primer} "
                     f"--p-r-primer {r_primer} "
                     f"--p-trunc-len {trunc_len} "
                     f"--p-trim-left {trim_left} "
                     f"--o-reads {project_path}/temp/trimmed_rep_file.qza")
    else:
        paragraph_list.append('No filtering or extracting of taxonomic reads were done as there were no primers '
                              'as the primers were not specified. ')
        ShellCommand(f'cp {project_path}/temp/rep_file.qza {project_path}/temp/trimmed_rep_file.qza')

    ShellCommand("qiime feature-classifier fit-classifier-naive-bayes "
                 f"--i-reference-reads {project_path}/temp/trimmed_rep_file.qza "
                 f"--i-reference-taxonomy {project_path}/temp/tax_file.qza "
                 f"--o-classifier {project_path}/taxonomic_classifier/{database_name}_classifier.qza")

    paragraph_list.append([f"Reference sequences from {database_name} were trimmed using forward ({f_primer}) "
                 f"'and reverse ({r_primer}) primers. Additional filtering was applied to match those used in the "
                 f"denoising step (truncation to length: {trunc_len} and trimmed {trim_left} basepairs from the left"
                 f"side). Naive-bayes taxonomic classifier was trained with the q2-feature-classifier -plugin. "])

    para_writer(paragraph_list, 'taxonomy')

    return
