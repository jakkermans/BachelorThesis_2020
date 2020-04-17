import sys
from os import listdir  # to read files
from os.path import isfile, join, normpath # to read files
import pathlib
import csv
import conllu
from __init__ import FileReader


def get_filenames_in_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]  # Return a list of files in a certain folder

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    filereader = FileReader()
    parses = args[0]
    parse_files = get_filenames_in_folder(parses)

    #To parse all the conllu data
    filereader.parse_conllu_data(parse_files)
    #with open('processed_output_v2.txt', 'r', encoding='utf-8') as input_file:
    #    data = input_file.readlines()
    #    test_line = data[0]
    #    test_dict = {}
    #    test_sub_dict = {}
    #    split_testline = test_line.rstrip().split()
    #    for item in split_testline[1:]:
    #        item_info = item.strip('()').split(',')
    #        test_sub_dict[item_info[0]] = item_info[1]
    #    test_dict[split_testline[0]] = test_sub_dict
    #    print(test_dict)

if __name__ == "__main__":
    main()