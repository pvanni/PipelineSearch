import subprocess
import pandas as pd
import numpy as np
import os
import shutil


class ShellCommand:
    """Takes shell command and run it"""

    def __init__(self, comstring):
        self.runcommand(comstring)

    def runcommand(self, command):
        process = subprocess.Popen(command.split('\n'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        process.wait()

        stdout, stderr = process.communicate()

        if process.returncode > 0:
            print(f"{stdout.decode('utf-8')}")
            print(f"{stderr.decode('utf-8')}")
            raise Exception('Error')

        return stdout


def results_manager(target_variables=False):
    """Decorator factory to write result txt files and iterate through all target variables"""
    def results_decorator(func):
        def results_wrapper(*args, **kwargs):

            target_list = []
            if target_variables:
                meta = pd.read_csv(f"{kwargs['project_path']}/start_files/metadata.tsv", sep='\t')
                for column in meta.columns.values:
                    unique = np.unique(meta[column].values, return_counts=True)
                    if 3 >= len(unique[0]) > 1 and np.amin(unique[1]) > 1:
                        target_list.append(column)

                for target_column in target_list:
                    print(target_column)
                    para = func(target_column=target_column, *args, **kwargs)
                    if para:
                        with open(f"{kwargs['project_path']}/methods/{func.__name__}", 'w+') as f:
                            for line in para:
                                f.write(f'{line}\n')

            else:
                # This is a method function, so write the paragraphs into a methods folder
                para = func(*args, **kwargs)
                with open(f"{kwargs['project_path']}/methods/{func.__name__}", 'w+') as f:
                    for line in para:
                        f.write(f'{line}\n')

        return results_wrapper

    return results_decorator


def tempfile_cleaner(target_dir, dont_remove_list=None):
    """Goes into the directory and removes every folder and file except the specified ones"""
    if dont_remove_list is None:
        dont_remove_list = ['']

    files = os.listdir(target_dir)

    # Go through each file
    for file in files:

        if file not in dont_remove_list:
            # Test if the file is a directory
            if os.path.isdir(f'{target_dir}/{file}'):
                # Removes recursively the whole directory and its contents
                shutil.rmtree(f'{target_dir}/{file}')
                print(f'Removed: {target_dir}/{file}')
            else:
                os.remove(f'{target_dir}/{file}')
                print(f'Removed: {target_dir}/{file}')

    return


def para_writer(para_list, name):

    with open(f'{name}.txt', 'w+') as f:
        for line in para_list:
            f.write(f'{line}\n')

    return


def restart(project_path, save_files=None):
    """Remove all files and restore the whole directory to initial order"""

    if save_files is None:
        save_files = ['']

    save_files += ['start_files', 'taxonomic_classifier', 'sequences', 'PipelineSearch', 'analyze.py', 'taxonomic_database', 'temp', 'batch_job.sh']
    tempfile_cleaner(target_dir=project_path, dont_remove_list=save_files)
    tempfile_cleaner('temp')
    tempfile_cleaner('taxonomic_classifier')

    return


def return_negative_rep_seqs(input_path):
    # Used only in special cases

    ShellCommand(f'qiime tools export '
                 f'--input-path {input_path} '
                 f'--output-path temp/neg_seqs')

    ShellCommand('seqtk seq -r temp/neg_seqs/dna-sequences.fasta > temp/reversed.fastq')

    ShellCommand(f'rm {input_path}')

    ShellCommand('qiime tools import '
                 '--input-path temp/reversed.fastq '
                 "--type 'FeatureData[Sequence]' "
                 f"--output-path {input_path}")
