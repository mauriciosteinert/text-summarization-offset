#!/usr/bin/python3


import argparse
import os



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

    sentences = text.lower().split("\n")
    # text = text.replace("\n", "")
    print("------------------------------------------------------------------------------")
    print(sentences)

    return [], ""



def main():
    parser = parse_args()

    file_list = os.listdir(parser.dataset_dir)

    if parser.process_n_examples != None:
        file_list = file_list[:int(parser.process_n_examples)]

    for file in file_list:
        sentences, ground_truth = preprocess_text(parser.dataset_dir + "/" + file)


if __name__ == "__main__":
    main()
