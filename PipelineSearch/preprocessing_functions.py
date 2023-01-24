from PipelineSearch.util import ShellCommand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PipelineSearch.util import para_writer


def import_function(manifest_type=True, paired=False, error_rate=0, sample_column='#SampleID'):

    """Takes in either multiplexed or demultiplexed formats

    Requirements in manifest_type=True:
    There needs to be a metadata file where #SampleID is the column name for identifiers in start_files called
    "metadata.tsv".
    Each sample must have an {id}.fastq or {id}.fastq.gz file in the .../sequences directory
    If theres no fastq.gz or fastq file for any given sample found in the metadata, then the sample is dropped

    If manifest_type=False, then its assumed for sequences to be Single End demultiplexed format in a single file
    with a barcode sequence. Barcodes are assumed to be found in BarcodeSequence column"""

    paragraph_list = []
    project_path = os.getcwd()
    path_to_sequences = os.getcwd() + '/sequences'

    # Some of the qiime commands require that there is a specifically called "sample id" column as the first column
    meta = pd.read_csv(f'{project_path}/start_files/start_metadata.tsv', sep='\t')
    columns = list(meta.columns.values)
    columns.remove('#SampleID')
    if sample_column == "#SampleID":
        meta = meta[['#SampleID'] + columns]
    else:
        meta['#SampleID'] = meta[sample_column].copy()
        meta = meta[['#SampleID'] + columns]

    if manifest_type:

        sample_names = list(meta[sample_column].values)

        file_path = path_to_sequences

        # Create the absolute filepath variable
        no_file_list = []
        file_path_list = []
        reverse_path_list = []
        if paired:
            for i in sample_names:
                for orientation in [1, 2]:
                    if os.path.isfile(f'{file_path}/{i}_{orientation}.fastq.gz'):
                        ShellCommand(f'gunzip {file_path}/{i}_{orientation}.fastq.gz')
                    if os.path.isfile(f'{file_path}/{i}_{orientation}.fastq'):
                        if orientation == 1:
                            file_path_list.append(f'{file_path}/{i}_{orientation}.fastq')
                        if orientation == 2:
                            reverse_path_list.append(f'{file_path}/{i}_{orientation}.fastq')
                    else:
                        no_file_list.append(i)
        else:
            for i in sample_names:
                if os.path.isfile(f'{file_path}/{i}.fastq.gz'):
                    ShellCommand(f'gunzip {file_path}/{i}.fastq.gz')
                if os.path.isfile(f'{file_path}/{i}.fastq'):
                    file_path_list.append(f'{file_path}/{i}.fastq')
                else:
                    no_file_list.append(i)

        meta.index = meta[sample_column].values

        # Report which samples were not found as fastq files and how many there were at the start
        paragraph_list.append(f'There were {meta.shape[0]} samples '
                              f'in the original metadata and {meta.shape[1]} columns. '
                              f'{len(no_file_list)} samples didnt have a fastq file in the sequences/ directory. '
                              f'The following sample IDs were removed from metadata.tsv:')

        paragraph_list.append(f'{no_file_list}')

        meta.drop(index=no_file_list, inplace=True)

        if paired:
            manifest = pd.DataFrame(data={'sample-id': meta[sample_column].values,
                                          'forward-absolute-filepath': file_path_list,
                                          'reverse-absolute-filepath': reverse_path_list},
                                    columns=['sample-id', 'forward-absolute-filepath', 'reverse-absolute-filepath'])
        else:
            manifest = pd.DataFrame(data={'sample-id': meta[sample_column].values, 'absolute-filepath': file_path_list},
                                    columns=['sample-id', 'absolute-filepath'])

        manifest.to_csv(f'{project_path}/temp/manifest.tsv', index=False, header=True, sep='\t')
        meta.to_csv(f'{project_path}/start_files/metadata.tsv', index=False, header=True, sep='\t')

        if paired:
            ShellCommand('qiime tools import '
                         "--type 'SampleData[PairedEndSequencesWithQuality]' "
                         f"--input-path {project_path}/temp/manifest.tsv "
                         f"--output-path {project_path}/temp/demultiplexed_sequences.qza "
                         "--input-format PairedEndFastqManifestPhred33V2")
        else:
            ShellCommand('qiime tools import '
                         "--type 'SampleData[SequencesWithQuality]' "
                         f"--input-path {project_path}/temp/manifest.tsv "
                         f"--output-path {project_path}/temp/demultiplexed_sequences.qza "
                         "--input-format SingleEndFastqManifestPhred33V2")

        paragraph_list.append("Sequences were imported into Qiime 2 ")

    else:
        # Compress
        ShellCommand(f'gzip {path_to_sequences}/sequences.fastq')

        # Import
        ShellCommand('qiime tools import '
                     '--type MultiplexedSingleEndBarcodeInSequence '
                     f'--input-path {project_path}/sequences.fastq.gz '
                     f'--output-path {project_path}/sequences/sequences.qza')

        paragraph_list.append("Multiplexed sequences were imported into Qiime2. ")

        ShellCommand('qiime cutadapt demux-single '
                     '--i-seqs sequences/sequences.qza '
                     f'--m-barcodes-file {project_path}/start_files/metadata.tsv '
                     '--m-barcodes-column BarcodeSequence '
                     f'--p-error-rate {error_rate} '
                     f'--output-dir {project_path}/temp/demultiplexed_sequences.qza')

        paragraph_list.append("Barcode sequences were removed using the q2-cutadapt -plugin. "
                              f"Error rate parameter was set to {error_rate}, while "
                              "other parameters were set to default values. ")

    para_writer(paragraph_list, 'import')
    return


def filter_preclustering(phred_threshold=25, lower_limit=100, paired=False, join_sequences=False):

    if paired:
        mode = 'paired'
    else:
        mode = 'single'

    # Remove sequences shorter than the limit
    ShellCommand(f'qiime cutadapt trim-{mode} ' 
                 '--i-demultiplexed-sequences temp/trimmed_demultiplexed_sequences.qza '
                 f'--p-minimum-length {lower_limit} '
                 f'--o-trimmed-sequences temp/qfilt_demultiplexed_sequences.qza')
    # Remove sequences with the average PHRED score of the threshold or lower
    ShellCommand(f'qiime quality-filter q-score '
                 '--i-demux temp/qfilt_demultiplexed_sequences.qza '
                 f'--p-min-quality {phred_threshold} '
                 '--o-filtered-sequences temp/qfilt_demultiplexed_sequences.qza '
                 f'--o-filter-stats temp/qfilt_stats.qza')

    """
    if paired:
        ShellCommand('qiime vsearch join-pairs '
                     '--i-demultiplexed-seqs temp/qfilt_demultiplexed_sequences.qza '
                     '--o-joined-sequences temp/qfilt_demultiplexed_sequences.qza')
    """

    # Dereplicate
    ShellCommand('qiime vsearch dereplicate-sequences '
                 '--i-sequences temp/qfilt_demultiplexed_sequences.qza '
                 '--o-dereplicated-table temp/qfilt_table.qza '
                 '--o-dereplicated-sequences temp/qfilt_sequences.qza')

    return


def primer_removal(forward_list, reverse_list=None, error_rate=0, threshold=0.95, no_trunc=False, paired=False):

    """Remove primers and create a report + statistics showing how many sequences got shorter"""
    paragraph_list = []
    picture_list = []
    project_path = os.getcwd()

    if forward_list:
        if not os.path.isfile(f'sequences/trimmed_demultiplexed_sequences.qza'):
            ShellCommand(f'cp {project_path}/temp/demultiplexed_sequences.qza '
                         f'{project_path}/temp/trimmed_demultiplexed_sequences.qza')
            if paired:
                for forward, reverse in zip(forward_list, reverse_list):
                    ShellCommand('qiime cutadapt trim-paired '
                                 f'--i-demultiplexed-sequences {project_path}/temp/trimmed_demultiplexed_sequences.qza '
                                 '--p-cores 4 '
                                 f'--p-front-f {forward} '
                                 f'--p-front-r {reverse} '
                                 f'--p-error-rate {error_rate} '
                                 f'--o-trimmed-sequences {project_path}/temp/trimmed_demultiplexed_sequences.qza')
            else:
                for forward in forward_list:
                    ShellCommand('qiime cutadapt trim-single '
                                 f'--i-demultiplexed-sequences {project_path}/temp/trimmed_demultiplexed_sequences.qza '
                                 '--p-cores 4 '
                                 f'--p-front {forward} '
                                 f'--p-error-rate {error_rate} '
                                 f'--o-trimmed-sequences {project_path}/temp/trimmed_demultiplexed_sequences.qza')

            paragraph_list.append(f"Primer sequences (f: {forward_list}, r: {reverse_list}) were removed using the "
                                  f"q2-cutadapt -plugin with "
                                  f"error rate parameter set to {error_rate}. Other parameters were set to default values. ")

    else:
        ShellCommand(f'cp {project_path}/temp/demultiplexed_sequences.qza '
                     f'{project_path}/temp/trimmed_demultiplexed_sequences.qza')

        paragraph_list.append(f"No primers were removed. ")

    # Export the sequences before primer trimming
    if not os.path.isdir(f'{project_path}/temp/demultiplexed_sequences_fastq/'):
        ShellCommand('qiime tools export '
                     f'--input-path {project_path}/temp/demultiplexed_sequences.qza '
                     f'--output-path {project_path}/temp/demultiplexed_sequences_fastq')

    if not os.path.isdir(f'{project_path}/temp/trimmed_demultiplexed_sequences_fastq/'):
        # Export the sequences after primer trimming
        ShellCommand('qiime tools export '
                     f'--input-path {project_path}/temp/trimmed_demultiplexed_sequences.qza '
                     f'--output-path {project_path}/temp/trimmed_demultiplexed_sequences_fastq')

    def export_for_quality(before_after):
        # List of files on the "before" directory
        file_before = os.listdir(f'{project_path}/temp/{before_after}demultiplexed_sequences_fastq/')
        for file in file_before:
            if 'fastq.gz' in file:
                ShellCommand(f'gunzip {project_path}/temp/{before_after}demultiplexed_sequences_fastq/{file}')

        # Create a list of fastq files in the directory
        file_before = os.listdir(f'{project_path}/temp/{before_after}demultiplexed_sequences_fastq/')
        before_fastq_list = []
        for file in file_before:
            if 'fastq' in file:
                before_fastq_list.append(f'{file}')

        return before_fastq_list

    # generate a list of fastq files for both before and after trimming
    before_list = export_for_quality('')
    after_list = export_for_quality('trimmed_')

    print(f'Are the before and after fastq lists the same: {before_list == after_list}')

    def sequence_parser(file_list, path):
        """Parse all the sequences to gather information for quality control"""
        fastq_data = []
        for file_path in file_list:
            name_list = []
            sequence_list = []
            quality_list = []
            length_list = []
            with open(f'{path}{file_path}') as file:
                counter = 1
                for new_line in file:
                    line = new_line.replace("\n", "")
                    if counter == 1:
                        name_list.append(line)
                    if counter == 2:
                        sequence_list.append(line)
                        length_list.append(len(line))
                    if counter == 4:
                        quality_list.append(line)
                        counter = 1
                        continue
                    counter += 1
            fastq_data.append([name_list, sequence_list, quality_list, length_list])

        return fastq_data

    def length_analyzer(fastq_data, max_length, threshold=0.95):
        """Determine how many samples survive just the truncation process
        for denoising. Assume that samples with over 2000 sequences survive
        after denoising. Maximize length, but keep atleast 95% of the samples"""

        # Contains how many sequences survive per sample for each length
        trunc_len_bins = [[] for i in range(100, max_length)]
        for sample in fastq_data:
            for i, value in enumerate(range(100, max_length)):
                trunc_len_bins[i].append((np.array(sample[3]) >= value).sum())

        n_samples = len(fastq_data)
        acceptable_lengths = []
        for length in trunc_len_bins:
            if (np.array(length) >= 4000).sum() >= n_samples * threshold:
                acceptable_lengths.append(True)
            else:
                acceptable_lengths.append(False)

        chosen_length = 0
        for i, value in enumerate(range(100, max_length)):
            if acceptable_lengths[i]:
                chosen_length = value

        return chosen_length

    # A list of fastq sequences with quality in the same order as the input list of fastq file paths were
    if len(before_list) > 50:
        before_data = sequence_parser(before_list[:10], f'{project_path}/temp/demultiplexed_sequences_fastq/')
        after_data = sequence_parser(after_list[:10], f'{project_path}/temp/trimmed_demultiplexed_sequences_fastq/')
    else:
        before_data = sequence_parser(before_list, f'{project_path}/temp/demultiplexed_sequences_fastq/')
        after_data = sequence_parser(after_list, f'{project_path}/temp/trimmed_demultiplexed_sequences_fastq/')


    def primer_trim_analyzer(before_input, after_input):
        # Take one sample at a time in the outer loop
        bp_removed = []

        for before, after in zip(before_data, after_data):
            # Assume that all sequences are in the same order and intact after filtering
            for before_seq, before_name, after_seq, after_name in zip(before[1], before[0], after[1], after[0]):
                if before_name == after_name:
                    bp_removed.append(len(before_seq) - len(after_seq))
        return bp_removed

    # The average amount of bp removed in
    bp_removed = primer_trim_analyzer(before_data, after_data)
    paragraph_list.append(f"On average {np.mean(bp_removed)} basepairs ({len(bp_removed)} sequences sampled) "
                          f"were removed from samples with q2-cutadapt. ")

    all_quality_scores = []
    for sample in after_data:
        for quality in sample[2]:
            all_quality_scores.append(quality)

    quality_sample = np.random.choice(all_quality_scores, size=10000, replace=False)
    quality_sample_scores = []
    max_len = 0
    # Transform to quality scores
    for sample in quality_sample:
        q_scores = []
        # Transfer the ascii symbols to PHRED33 quality scores
        for letter in sample:
            q_scores.append(ord(letter) - 33)
        # Record the max lenght needed for plotting
        if len(q_scores) > max_len:
            max_len = len(q_scores)
        quality_sample_scores.append(q_scores)

    # Bin the data into each boxplot
    box_data = [[] for i in range(max_len)]

    for sample in quality_sample_scores:
        for i, value in enumerate(sample):
            box_data[i].append(value)

    # plot the data. Find out in what format the picture needs to be returned for the report
    fig, ax = plt.subplots(figsize=(20.0, 10.0))
    ax.boxplot(box_data, showfliers=False, showcaps=False)
    chosen_length = length_analyzer(after_data, max_len, threshold)
    ax.axvline(chosen_length, color='r')
    ax.set_xticks([i for i in np.arange(0, max_len, 20)])
    ax.set_xticklabels([i for i in np.arange(0, max_len, 20)])

    fig.savefig(f"{project_path}/sequence_quality.png")
    plt.close('all')
    picture_list.append(f"{project_path}/sequence_quality.png")

    if no_trunc:
        print(f"No truncation for denoising were chosen because you selected the parameter no_trunc. ")
        paragraph_list.append(f"No truncation for denoising were chosen. ")
        para_writer(paragraph_list, 'primer_trimming')
        return 0
    else:
        paragraph_list.append(f"For denoising {chosen_length} was initially chosen. ")
        para_writer(paragraph_list, 'primer_trimming')
        return chosen_length


def dada2(trunc_len, trim_left, speed=5, paired=False, threshold=0.95):
    """Denoise the data based on the trunc len found in primer_removal function.
    If more than 10% of the samples are lost during denoising, reduce trunc_len size by 5 and repeat the process"""
    paragraph_list = []
    project_path = os.getcwd()

    def dada2_denoising(trunc_len, trim_left, paired=False):
        """Run the dada2"""

        if not paired:
            if not os.path.isdir(f'{project_path}/temp/{trunc_len}_dada2_output/'):
                ShellCommand("qiime dada2 denoise-single "
                             f"--i-demultiplexed-seqs {project_path}/temp/trimmed_demultiplexed_sequences.qza "
                             f"--p-trim-left {trim_left} "
                             f"--p-trunc-len {trunc_len} "
                             f"--p-n-threads 1 "
                             f"--output-dir {project_path}/temp/{trunc_len}_dada2_output")
        else:
            if not os.path.isdir(f'{project_path}/denoising/{trunc_len}_dada2_output/'):
                ShellCommand("qiime dada2 denoise-paired "
                             f"--i-demultiplexed-seqs {project_path}/temp/trimmed_demultiplexed_sequences.qza "
                             f"--p-trim-left-f {trim_left} "
                             f"--p-trunc-len-f {trunc_len} "
                             f"--p-trim-left-r {trim_left} "
                             f"--p-trunc-len-r {trunc_len} "
                             f"--p-n-threads 1 "
                             f"--output-dir {project_path}/temp/{trunc_len}_dada2_output")

    def analyze_dada2_report(trunc_len, speed=5, threshold=0.95):
        """Export the denoising stats and determine which direction to go"""

        if not os.path.isdir(f"{project_path}/temp/{trunc_len}_dada2_output/stats/"):
            ShellCommand("qiime tools export "
                         f"--input-path {project_path}/temp/{trunc_len}_dada2_output/denoising_stats.qza "
                         f"--output-path {project_path}/temp/{trunc_len}_dada2_output/stats")

        df = pd.read_csv(f'{project_path}/temp/{trunc_len}_dada2_output/stats/stats.tsv', sep='\t')
        df.drop(index=0, inplace=True)
        df.drop(columns=['sample-id'], inplace=True)
        n_samples = df.shape[0]
        df = df.astype(float)

        length = df['non-chimeric'].values

        paragraph = [f"After denoising step {(length >= 1000).sum()} samples had 1000 "
                     f"or more sequences with {trunc_len} trunc_len parameter "
                     f"({n_samples} before denoising). "]

        if trunc_len == 0:
            return speed, paragraph, True
        if (length >= 1000).sum() >= n_samples * threshold:
            return speed, paragraph, True
        else:
            return -speed, paragraph, False

    # Export the dada2 report and infer how many samples had under 1000 sequences after. Reduce the trunc_len and
    # repeat it again. Only do it 10 times and the abort

    paragraph_list.append("Sequences were denoised to ASV's using the q2-dada2 -plugin. ")

    # If trunc len is 0 at the start, only do one run
    if trunc_len == 0:
        redo = 9
    else:
        redo = 0

    length_change = 0
    already_tried = []
    chosen_trunc_len = 0
    while redo < 10:
        try:
            dada2_denoising(trunc_len + length_change, trim_left, paired=paired)
        except:
            if already_tried == []:
                raise Exception('Error in the dada2 in the first try. Examine')
            else:
                print('Some attempt passed. Continue')
                break

        # Analyze and try with a new length
        new_length, paragraph, good_length = analyze_dada2_report(trunc_len + length_change, speed, threshold)

        # Write paragraphs and update the chosen_trunc_len
        paragraph_list.append(paragraph)
        if good_length:
            chosen_trunc_len = trunc_len + length_change
            paragraph_list.append(f"Truncation length {chosen_trunc_len} passed the {threshold} threshold. ")
        else:
            paragraph_list.append(f"Truncation length {trunc_len + length_change} failed the {threshold} threshold. ")

        already_tried.append(length_change)

        # Change the length for the next iteration
        length_change += new_length
        # Dont go backwards for no reason, come on man
        if length_change in already_tried:
            redo = 10
        else:
            redo += 1

    # Incase of total failure to find any trunc_len to save the current threshold
    if chosen_trunc_len == 0:
        paragraph_list.append("Did not found any appropriate trunc_len with the chosen threshold")

    para_writer(paragraph_list, 'dada2')

    # Create the dada2 folder and put in the selected feature_table and representative sequences inside
    os.mkdir('dada2')

    ShellCommand(f'cp {project_path}/temp/{chosen_trunc_len}_dada2_output/table.qza dada2/feature_table.qza')
    ShellCommand(f'cp {project_path}/temp/{chosen_trunc_len}_dada2_output/representative_sequences.qza dada2/representative_sequences.qza')

    return chosen_trunc_len


def denovo_clustering(perc_identity, threads=1):

    os.mkdir(f'denovo')

    ShellCommand('qiime vsearch cluster-features-de-novo '
                 '--i-sequences temp/qfilt_sequences.qza '
                 '--i-table temp/qfilt_table.qza '
                 f'--p-perc-identity {perc_identity} '
                 f'--p-threads {threads} '
                 f'--o-clustered-table denovo/feature_table.qza '
                 f'--o-clustered-sequences denovo/representative_sequences.qza')

    return


def closed_clustering(database, ref_seqs, perc_identity, threads=1):

    os.mkdir(f'closed_{database}')

    ShellCommand("qiime tools import "
                 "--type 'FeatureData[Sequence]' "
                 f"--input-path {ref_seqs} "
                 f"--output-path temp/{database}_ref_seqs.qza")

    ShellCommand('qiime vsearch cluster-features-closed-reference '
                 '--i-sequences temp/qfilt_sequences.qza '
                 '--i-table temp/qfilt_table.qza '
                 f'--i-reference-sequences temp/{database}_ref_seqs.qza '
                 f'--p-perc-identity {perc_identity} '
                 f'--p-threads {threads} '
                 f"--p-strand 'both' "
                 f'--o-clustered-table closed_{database}/feature_table.qza '
                 f'--o-clustered-sequences closed_{database}/representative_sequences.qza '
                 f'--o-unmatched-sequences closed_{database}/unmatched_sequences.qza')

    return