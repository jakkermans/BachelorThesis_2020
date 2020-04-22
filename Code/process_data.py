import sys
from os import listdir  # to read files
from os.path import isfile, join, normpath # to read files
import os
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


if __name__ == "__main__":
    main()