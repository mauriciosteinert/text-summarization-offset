#!/usr/bin/python3


import sys
sys.path.append("../pucrs-cc-tcc-2018/lib/python/")
sys.path.append("../pucrs-cc-tcc-2018/lib/rouge/")

import argparse
import os
import sent2vec
import rouge


rouge = rouge.Rouge()
sent2vec_model = sent2vec.Sent2vecModel()



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
    sentences = []
    while full_text:
        sentence = full_text.pop(0)

        if sentence == "@highlight":
            break
        sentences.append(sentence)

    # Concatenate ground-truth value
    ground_truth = ""

    while full_text:
        sentence = full_text.pop(0)

        if sentence == "@highlight":
            continue

        ground_truth += sentence + ". "

    # Discard sentences shorther than 30 characters
    sentences = [sentence for sentence in sentences if len(sentence) >= 30]

    return sentences, ground_truth



def process_summary(sentences, ground_truth):
    # Compute ROUGE scores for each sentence

    for sentence in sentences:
        try:
            rouge_str = rouge.get_scores(ground_truth, sentence)
        except ValueError:
            log_file = open(parser.log_file_name + "-error", "a")
            log_file.write("[" + file + "] Error computing ROUGE scores for sentence " + sentence + "\n")
            log_file.close()
            continue




def main():
    parser = parse_args()

    file_list = os.listdir(parser.dataset_dir)

    if parser.process_n_examples != None:
        file_list = file_list[:int(parser.process_n_examples)]

    total_files = len(file_list)
    file_idx = 0

    for file in file_list:
        print("[" + str(file_idx / total_files) + "] Processing file " + file)
        file_idx += 1

        sentences, ground_truth = preprocess_text(parser.dataset_dir + "/" + file)

        # print("\n------------------------------------------------------------------------------")
        # print("Sentences = ", sentences)
        # print("Ground truth = ", ground_truth)

        # Ignore texts shorter than 15 sentences
        if len(sentences) < 10 or len(ground_truth) < 20:
            log_file = open(parser.log_file_name + "-error", "a")
            log_file.write("Ignoring file " + file + " due to short length.\n")
            log_file.close()
            continue

        process_summary(sentences, ground_truth)

if __name__ == "__main__":
    main()
