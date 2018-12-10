#!/usr/bin/python3
#
# Vector Offset Summarization: an extractive text summarization method that uses
# word embeddings arithmetic to evaluate sentences similarity, selecting as summary
# n sentences that are close to whole text representation.
#
# Author: Mauricio Steinert
#

import rouge
import sent2vec
import numpy as np
import argparse
import os
import nltk
import matplotlib.pyplot as plt


# Parse input parameters
parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--process-n-examples', metavar='process_n_examples',
                    help='Total of examples to process from data set directory')
parser.add_argument('--dataset-dir', metavar='dataset_dir',
                    help='Data set directory with examples from CNN/Dailyail')
parser.add_argument('--word-vector-dictionary', metavar='word_vector_dictionary',
                    help='Word vector dictionary to use')
parser.add_argument('--max-summary-length', metavar='max_summary_length',
                    help='Maximum number of characters in summary')
parser.add_argument('--log-file-name', metavar='log_file_name',
                    help='Output file name')
parser.add_argument('--generate-tsne', metavar='generate_tsne',
                    help='Generate T-SNE visualization for each example')
parser = parser.parse_args()


# Load ROUGE class
rouge = rouge.Rouge()

# Load sent2vec model
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model(parser.word_vector_dictionary)

# Convert ROUGE string output to list
def rouge_to_list(rouge_str):
    rouge_list = [ [rouge_str[0]['rouge-1']['f'], rouge_str[0]['rouge-1']['p'], rouge_str[0]['rouge-1']['r'] ],\
                [rouge_str[0]['rouge-2']['f'], rouge_str[0]['rouge-2']['p'], rouge_str[0]['rouge-2']['r'] ],\
                [rouge_str[0]['rouge-l']['f'], rouge_str[0]['rouge-l']['p'], rouge_str[0]['rouge-l']['r'] ]\
                ]
    return rouge_list



# Return sentence and ground-truth value
def preprocess_text(text):
    sentences_all = text.lower().split("\n")
    sentences_all = [sentence.split(". ") for sentence in sentences_all]
    sentences = []
    ground_truth_ret = []


    for sentence in sentences_all:
        if type(sentence) == list:
            for subsentence in sentence:
                sentences.append(subsentence)
        else:
            sentences.append(sentence)


    sentences_ret = []
    while sentences:
        sentence = sentences.pop(0)

        if sentence == "@highlight":
            break

        sentences_ret.append(sentence)


    while sentences:
        sentence = sentences.pop(0)
        if sentence == "@highlight":
            continue

        ground_truth_ret.append(sentence)

    sentences_ret = [sentence for sentence in sentences_ret if len(sentence) > 30]
    ground_truth_ret = [sentence for sentence in ground_truth_ret if len(sentence) > 30]

    ground_truth_str = ""
    for sentence in ground_truth_ret:
        ground_truth_str = ground_truth_str + sentence + ". "

    return sentences_ret, ground_truth_str


# Process each file example
def process_example(filename):
    text = open(parser.dataset_dir + "/" + filename).read()

    sentences, ground_truth = preprocess_text(text)

    sentences_vec = sent2vec_model.embed_sentences(sentences)
    ground_truth_vec = sent2vec_model.embed_sentence(ground_truth)

    # Compute mean vector of whole text
    text_mean_vec = np.mean(sentences_vec, axis=0)

    # Extract ground-truth vector from whole text
    text_mean_diff_vec = np.subtract(text_mean_vec, ground_truth_vec)

    sentence_idx = 0
    sentences_dist = []

    for sentence_vec in sentences_vec:
        sentence_vec_dist = np.linalg.norm((text_mean_vec, np.add(sentence_vec, text_mean_diff_vec)))

        try:
            rouge_str = rouge.get_scores(ground_truth, sentences[sentence_idx])
        except ValueError:
            sentence_idx += 1
            continue

        sentences_dist.append([sentence_idx, sentence_vec_dist,
                                rouge_to_list(rouge_str)])
        sentence_idx += 1

    # Sort sentences based on closest vector distance
    sentences_dist.sort(key=lambda x: x[1])
    best_vector_idx = sentences_dist[0][0]
    sentences_vector_idx = []
    best_vector_str = ""

    for sentence in sentences_dist:
        if len(best_vector_str) > 200:
            break
        best_vector_str += sentences[sentence[0]] + ". "
        sentences_vector_idx.append(sentence[0])


    # Sort based on best ROUGE-1 score
    sentences_dist.sort(key=lambda x: x[2][0], reverse=True)
    best_rouge_idx = sentences_dist[0][0]
    sentences_rouge_idx = []
    best_rouge_str = ""

    for sentence in sentences_dist:
        if len(best_rouge_str) > int(parser.max_summary_length):
            break
        best_rouge_str += sentences[sentence[0]] + ". "
        sentences_rouge_idx.append(sentence[0])

    log_file = open(parser.log_file_name, "a")

    log_file.write("\n\nProcessing " + filename)
    log_file.write("\n* GROUND TRUTH = " + ground_truth)
    log_file.write("\n* BEST VECTOR SUMMARY = " + best_vector_str)
    log_file.write("\n* BEST ROUGE SUMMARY = " + best_rouge_str)
    log_file.write("\n* ROUGE VECTOR SCORES = " + str(rouge_to_list(rouge.get_scores(ground_truth, best_vector_str))))
    log_file.write("\n* ROUGE BEST SCORES = " + str(rouge_to_list(rouge.get_scores(ground_truth, best_rouge_str))))
    log_file.close()

    best_vector_vec = sent2vec_model.embed_sentence(best_vector_str)

    if parser.generate_tsne == "True":
        sentences_vec_tsne = np.vstack((sentences_vec, text_mean_vec, ground_truth_vec, best_vector_vec))

        first_sentence = True
        U, s, Vh = np.linalg.svd(sentences_vec_tsne, full_matrices=False)

        for i in range(len(sentences_vec)):
            fig = plt.gcf()
            fig.set_size_inches(5, 5)

            if first_sentence == True:
                plt.plot(U[i, 0], U[i, 1], 'go', label='sentence',
                        markersize=16)
                first_sentence = False
            else:
                plt.plot(U[i, 0], U[i, 1], 'go', markersize=16)

        # Text mean
        plt.plot(U[len(sentences_vec), 0], U[len(sentences_vec), 1], 'bs', label='text mean', markersize=16)
        plt.plot(U[len(sentences_vec) + 1, 0], U[len(sentences_vec) + 1, 1], 'r^', label='ground truth', markersize=14)

        # Plot vector selected sentences
        first_sentence = True
        for i in sentences_vector_idx:
            if first_sentence == True:
                plt.plot(U[i, 0], U[i, 1], 'm+', label='vector selected sentences', markersize=24)
                first_sentence = False
            else:
                plt.plot(U[i, 0], U[i, 1], 'm+', markersize=24)

        # Plot best rouge selected sentences
        first_sentence = True
        for i in sentences_rouge_idx:
            if first_sentence == True:
                plt.plot(U[i, 0], U[i, 1], 'yx', label='rouge selected sentences', markersize=24)
                first_sentence = False
            else:
                plt.plot(U[i, 0], U[i, 1], 'yx', markersize=24)


        # Save TSNE file
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.05), ncol=4, prop={'size': 6})

        for leg_handle in legend.legendHandles:
            leg_handle._legmarker.set_markersize(6)

        plt.savefig("tsne/" + filename + ".png", format="png")
        plt.clf()

    return [rouge_to_list(rouge.get_scores(ground_truth, best_vector_str)),
            rouge_to_list(rouge.get_scores(ground_truth, best_rouge_str))
            ], best_vector_str, best_rouge_str





# Get files from data set directory
file_idx = 0
files_list = os.listdir(parser.dataset_dir)

# Remove old log file
try:
    os.remove(parser.log_file_name)
except FileNotFoundError:
    None

if parser.process_n_examples != None:
    files_list = files_list[:int(parser.process_n_examples)]

log_file = open(parser.log_file_name, "a")
log_file.write(str(parser))
log_file.close()

stats = []
summary_match = 0

total_examples = len(files_list)
file_step = total_examples / 1000

for file in files_list:
    if file_idx % file_step == 0:
        print("[" + str(file_idx / total_examples) + "]  Processing file " + file)

    try:
        stat, best_vector_str, best_rouge_str = process_example(file)
        stats.append(stat)
    except TypeError:
        log_file = open(parser.log_file_name + ".error", "a")
        log_file("Error processing file " + file)
        log_file.close()
        continue

    if best_vector_str == best_rouge_str:
        summary_match += 1
    file_idx += 1

stats_mean = np.mean(stats, axis=0)
stats_std = np.std(stats, axis=0)

log_file = open(parser.log_file_name, "a")
log_file.write("\n\n---------------------------------- DATA SET STATISTICS ----------------------------------")
log_file.write("\nTOTAL OF EXAMPLES: " + str(file_idx))
log_file.write("\nVECTOR OFFSET")
log_file.write("\n\tROUGE-1 mean: " + str(stats_mean[0][0][0]))
log_file.write("\n\tROUGE-2 mean: " + str(stats_mean[0][1][0]))
log_file.write("\n\tROUGE-L mean: " + str(stats_mean[0][2][0]))
log_file.write("\n\tROUGE-1 std: " + str(stats_std[0][0][0]))
log_file.write("\n\tROUGE-2 std: " + str(stats_std[0][1][0]))
log_file.write("\n\tROUGE-L std: " + str(stats_std[0][2][0]))

log_file.write("\n\nBEST ROUGE SCORES")
log_file.write("\n\tROUGE-1 mean: " + str(stats_mean[1][0][0]))
log_file.write("\n\tROUGE-2 mean: " + str(stats_mean[1][1][0]))
log_file.write("\n\tROUGE-L mean: " + str(stats_mean[1][2][0]))
log_file.write("\n\tROUGE-1 std: " + str(stats_std[1][0][0]))
log_file.write("\n\tROUGE-2 std: " + str(stats_std[1][1][0]))
log_file.write("\n\tROUGE-L std: " + str(stats_std[1][2][0]))
