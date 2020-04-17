import sys
from os import listdir  # to read files
from os.path import isfile, join, normpath # to read files
import pathlib
import csv
import conllu
from __init__ import FileReader
from gensim.models import FastText


def parse_ud(ud_file):
    print("##### Processing UD parsed sentences")
    with open(ud_file, 'r', encoding='utf-8') as input_file:
        data = input_file.readlines()
        line_dict = {}
        lines_list = []
        for line in data:
            line_sub_dict = {}
            line_sub_list = []
            split_line = line.rstrip().lower().split()
            if len(split_line[1:]) != 0:
                for item in split_line[1:]:
                    try:
                        item_info = item.strip('()').split(',')
                        if item_info[1] != 'PUNCT':
                            line_sub_dict[item_info[0]] = item_info[1]
                            line_sub_list.append(item_info[0])
                    except IndexError:
                        pass
                line_dict[split_line[0]] = line_sub_dict
                lines_list.append(line_sub_list)

    return line_dict, lines_list

def generate_word_embeddings(lines_list):
    print('##### Generating word embeddings')
    we_model = FastText(lines_list, size=4, window=3, min_count=1, workers=4, sg=1)
    return we_model

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    ud_dict, review_sentences = parse_ud(args[0])
    we_model = generate_word_embeddings(review_sentences)
    print(we_model.wv.similarity('schrijven', 'stijl'))

if __name__ == "__main__":
    main()