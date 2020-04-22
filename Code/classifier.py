import sys
import os
from __init__ import FileReader
from gensim.models import FastText
import json
from nltk.tokenize import word_tokenize


def parse_ud(ud_file):
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
                        if item_info[1] != 'punct':
                            line_sub_dict[item_info[0]] = item_info[1]
                            line_sub_list.append(item_info[0])
                    except IndexError:
                        pass
                line_dict[split_line[0]] = line_sub_dict
                lines_list.append(line_sub_list)

    return line_dict, lines_list

def generate_word_embeddings():
    lines_list = []
    with open('final_output_v2.txt', 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        for line in lines:
            line = line.rstrip()[:-1]
            lines_list.append(line.split())

    we_model = FastText(lines_list, size=100, window=10, min_count=2, workers=10, sg=0, negative=5, iter=5)
    return we_model

def generate_feature_lexicon(data_dict, ud_dict, we_model):
    test_cases = [key for key in data_dict.keys()][0:50]
    feature_lexicon = {'1': {'stijl', 'taal', 'toon', 'ondertoon', 'penvoering', 'zinnetjes', 'zinnen', 'toonzetting', 'woorden', 'woordspelingen', 'woordenstroom', 'formuleringen', 'taalgebruik', 'vocabulaire', 'verteltrant', 'schrijfwijze', 'taalbeheersing', 'woordkeus', 'schrijven', 'beschrijven', 'weergeven', 'vertellen', 'benoemen', 'formuleren', 'op papier zetten', 'noteren', 'structuur', 'orde', 'ordeloosheid', 'sprongen in ruimte of tijd', 'vermenging', 'fragmentarisch', 'hoofdstukken', 'alinea\'s', 'lijnen', 'hoofdlijnen', 'zijlijntjes', 'vooruitwijzingen', 'flash-backs', 'patronen', 'perspectiefwisselingen', '\'rond\' verhaal', 'compositie', 'vervlechten', 'in elkaar zetten', 'componeren', 'verknopen', 'onderbreken', 'verbinden', 'vermengen'},
                       '2': {'plot', 'verhaal', 'verhaaldraad', 'slot', 'ontknoping', 'historie', 'handelingen', 'episoden', 'voorvallen', 'thema', 'idee', 'probleem', 'problematiek', 'gedachte', 'romanthese', 'visie', 'inzicht'},
                       '3': {'personages', 'figuur', 'hoofdpersoon', 'karakter', 'mens', 'tegenspeler', 'hoofdrol', 'bijrol', 'persoonlijkheid', 'sujet', 'held', 'type', 'dialoog', 'gesprek', 'woordenwisseling', 'uitspraak', 'vertellen', 'spreken'},
                       '4': {'uitgever', 'titel', 'illustratie', 'afbeelding', 'foto', 'flaptekst', 'papier', 'verschijnen', 'presenteren'},
                       '5': {'boek', 'werk', 'roman', 'verhaal', 'vertelling', 'deel', 'hoofdstuk', 'slot'}}

    aspect_dict = {'1': ['stijl', 'structuur'],
                   '2': ['plot', 'thema'],
                   '3': ['karakters', 'dialoog'],
                   '4': 'verschijning',
                   '5': 'gehele werk'}

    for case in test_cases:
        try:
            review_data = data_dict[case]
            pos_data = ud_dict[case]
            for token in review_data['sentence']:
                similarity_scores = []
                try:
                    if pos_data[token] in ['noun', 'adj', 'verb', 'propn', 'pron']:
                        if token not in feature_lexicon[review_data['label1'][0]]:
                            for aspect in aspect_dict[review_data['label1'][0]]:
                                similarity_scores.append(we_model.wv.similarity(token, aspect))
                                if max(similarity_scores) > 0.7:
                                    feature_lexicon[review_data['label1'][0]].add(token)
                        else:
                            pass
                    else:
                        pass
                except KeyError:
                    pass
        except KeyError:
            pass

    return feature_lexicon

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    filereader = FileReader()

    print("##### Processing UD parsed sentences")
    ud_dict, review_sentences = parse_ud(args[0])

    print('##### Reading in review data')
    review_dict, data_dict = filereader.parse_review_data(args[1])

    print('##### Generating word embeddings')
    we_model = generate_word_embeddings()

    print('##### Generating feature lexicon')
    feature_lexicon = generate_feature_lexicon(data_dict, ud_dict, we_model)
    for key, value in feature_lexicon.items():
        print(key, value)

if __name__ == "__main__":
    main()