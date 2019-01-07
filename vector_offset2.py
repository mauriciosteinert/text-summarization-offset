#!/usr/bin/python3


import sys
sys.path.append("../pucrs-cc-tcc-2018/lib/python/")
sys.path.append("../pucrs-cc-tcc-2018/lib/rouge/")

import argparse
import os
import sent2vec
import rouge
import numpy as np


def rouge_to_list(rouge_str):
    rouge_list = [ [rouge_str[0]['rouge-1']['f'], rouge_str[0]['rouge-1']['p'], rouge_str[0]['rouge-1']['r'] ],\
                [rouge_str[0]['rouge-2']['f'], rouge_str[0]['rouge-2']['p'], rouge_str[0]['rouge-2']['r'] ],\
                [rouge_str[0]['rouge-l']['f'], rouge_str[0]['rouge-l']['p'], rouge_str[0]['rouge-l']['r'] ]\
                ]
    return rouge_list


def parse_args():
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
    return parser



def preprocess_text(filename):
    text = open(filename, "r").read()

    # Split using line break as delimiter
    sentences = text.lower().split("\n")

    # Combine paragraphs
    full_text = []
    paragraph = ""

    for sentence in sentences:
        if len(sentence) == 0:
            full_text.append(paragraph)
            paragraph = ""
            continue
        paragraph += sentence

        if sentence == sentences[-1]:
            full_text.append(paragraph)

    # Separate text from ground-truth
    sentences_text = []
    while full_text:
        sentence = full_text.pop(0)

        if sentence == "@highlight":
            break
        sentences.append(sentence)

    # Concatenate ground-truth value
    ground_truth_text = ""

    while full_text:
        sentence = full_text.pop(0)

        if sentence == "@highlight":
            continue

        ground_truth_text += sentence + ". "

    # Discard sentences shorther than 30 characters
    sentences_text = [sentence for sentence in sentences if len(sentence) >= 30]

    sentences_vec = sent2vec_model.embed_sentences(sentences_text)
    ground_truth_vec = sent2vec_model.embed_sentence(ground_truth_text)

    return sentences_text, sentences_vec, ground_truth_text, ground_truth_vec



def generate_summary(filename):
    sentences_text, sentences_vec, ground_truth_text, ground_truth_vec = preprocess_text(filename)
    stat = []
    global top_n_counter

    # print("Sentences = ", sentences_text)
    # print("Sentences vec = ", sentences_vec)
    # print("Ground truth = ", ground_truth_text)
    # print("Ground truth vec = ", ground_truth_vec)

    if len(sentences_text) < 10:
        return None

    # Compute text mean without ground-truth
    text_mean_vec = np.mean(np.vstack((sentences_vec, ground_truth_vec)), axis=0)

    # Extract ground-truth from mean
    text_mean_diff_vec = np.subtract(text_mean_vec, ground_truth_vec)


    sentence_idx = 0
    for sentence_vec in sentences_vec:
        # Compute sentence distance from text text mean
        sentence_vec_dist = np.linalg.norm((text_mean_vec, np.add(text_mean_diff_vec[0], sentence_vec)))
        print(text_mean_diff_vec)
        print(len(text_mean_diff_vec), len(sentence_vec))
        print("Len Add = ", len(np.add(text_mean_diff_vec, sentence_vec)))

        # Compute ROUGE scores for sentences
        try:
            rouge_str = rouge.get_scores(ground_truth_text, sentences_text[sentence_idx])
        except ValueError:
            stat.append([sentence_idx, sentence_vec_dist, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
            sentence_idx += 1
            continue

        stat.append([sentence_idx, sentence_vec_dist, rouge_to_list(rouge_str)])
        sentence_idx += 1

    # Compute vector summary
    stat.sort(key=lambda x: x[1])
    sentences_vector_idx = []
    best_vector_str = ""

    for sentence in stat:
        if len(best_vector_str) > int(parser.max_summary_length):
            break
        best_vector_str += sentences_text[sentence[0]] + ". "
        sentences_vector_idx.append(sentence[0])

    # Compute ROUGE summary
    stat.sort(key=lambda x: x[2][0], reverse=True)
    sentences_rouge_idx = []
    best_rouge_str = ""

    for sentence in stat:
        if len(best_rouge_str) > int(parser.max_summary_length):
            break
        best_rouge_str += sentences_text[sentence[0]] + ". "
        sentences_rouge_idx.append(sentence[0])

    log_file.write("-----------------------------------------------------------------\n")
    log_file.write("Processing file " + str(filename) + "\n")
    log_file.write("Best vector indexes = " + str(sentences_vector_idx) + "\n")
    try:
        log_file.write("ROUGE Scores = " + str(rouge_to_list(rouge.get_scores(ground_truth_text, best_vector_str))) + "\n")
    except ValueError:
        log_file_error.write("[" + filename + "] Error computing ROUGE score for summary vector\n")

    log_file.write("Vector summary = " + str(best_vector_str) + "\n")
    log_file.write("Best ROUGE indexes " + str(sentences_rouge_idx) + "\n")
    log_file.write("ROUGE summary = " + str(best_rouge_str) + "\n")
    try:
        log_file.write("ROUGE Scores = " + str(rouge_to_list(rouge.get_scores(ground_truth_text, best_rouge_str))) + "\n")
    except ValueError:
        log_file_error.write("[" + filename + "] Error computing ROUGE score for ROUGE vector\n")

    log_file.write("Ground truth summary = " + ground_truth_text);

    for sentence_idx in sentences_vector_idx:
        for idx in range(0, top_n):
            if sentence_idx in sentences_rouge_idx[:idx]:
                top_n_counter[idx] += 1

    try:
        stat_vec = rouge_to_list(rouge.get_scores(ground_truth_text, best_vector_str))
    except:
        stat_vec = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    try:
        stat_rouge = rouge_to_list(rouge.get_scores(ground_truth_text, best_rouge_str))
    except:
        stat_rouge = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    stats_vec.append(stat_vec)
    stats_rouge.append(stat_rouge)



def main():
    sent2vec_model.load_model(parser.word_vector_dictionary)

    file_list = os.listdir(parser.dataset_dir)

    if parser.process_n_examples != None:
        file_list = file_list[:int(parser.process_n_examples)]

    total_files = len(file_list)
    file_idx = 0

    log_file.write(str(parser))

    for file in file_list:
        print("[" + str(file_idx / total_files) + "] Processing file " + file)
        file_idx += 1

        generate_summary(parser.dataset_dir + "/" + file)

    vec_mean = np.mean(stats_vec, axis=0)
    vec_std = np.std(stats_vec, axis=0)
    rouge_mean = np.mean(stats_rouge, axis=0)
    rouge_std = np.std(stats_rouge, axis=0)

    log_file.write("-----------------------------------------------------------------\n")
    log_file.write("Vec mean = " + str(vec_mean) + "\n")
    log_file.write("Vec std = " + str(vec_std) + "\n")
    log_file.write("ROUGE mean = " + str(rouge_mean) + "\n")
    log_file.write("ROUGE std = " + str(rouge_std) + "\n")
    log_file.write("Common sentences = " + str(top_n_counter) + "\n")
    log_file.write("Total of processed files = " + str(len(stats_vec)))

    log_file.close()
    log_file_error.close()



if __name__ == "__main__":
    parser = parse_args()


    # Remove old log file
    try:
        os.remove(parser.log_file_name)
        os.remove(parser.log_file_name + "-error")
    except FileNotFoundError:
        None


    log_file = open(parser.log_file_name, "a")
    log_file_error = open(parser.log_file_name + "-error", "a")
    #common_sentences = 0
    top_n = 10
    top_n_counter = np.zeros(top_n)

    rouge = rouge.Rouge()
    sent2vec_model = sent2vec.Sent2vecModel()
    stats_vec = []
    stats_rouge = []
    main()
