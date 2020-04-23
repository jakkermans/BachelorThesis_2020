import sys
import os
from __init__ import FileReader
from gensim.models import FastText
import json
from nltk.tokenize import word_tokenize
import csv
import re
from random import shuffle
from sklearn.svm import SVC, LinearSVC
import nltk.classify


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
            sublines = line.split('.')
            for subline in sublines:
                lines_list.append(subline.split())

    we_model = FastText(lines_list, size=100, window=10, min_count=1, workers=10, sg=0, negative=5)
    return we_model

def generate_feature_lexicon(data_dict, ud_dict, we_model):
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

    for key, value in data_dict.items():
        try:
            review_data = value
            pos_data = ud_dict[key]
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

def generate_featuresets(lexicon, data_dict, urls, metadata_dict):
    featuresets = []
    for key, value in data_dict.items():
        features = {}
        id_match = re.match('(\d+)_', key)
        for token in value['sentence']:
            if token in lexicon[value['label1'][0]]:
                features[token] = 'True'
        review_url = urls[int(id_match.group(1))+1]
        try:
            nur_code = metadata_dict[review_url]['nur']
            features[nur_code] = 'True'
        except KeyError:
            features['None'] = 'True'
        featuresets.append((features, value['label1'][0]))

    return featuresets

def split_feats(feats, split=0.6):
    train_feats = []
    test_feats = []

    shuffle(feats)
    cutoff = int(len(feats) * split)
    train_feats, test_feats = feats[:cutoff], feats[cutoff:]

    return train_feats, test_feats

def classification(train_feats, test_feats):
    svm_classifier = nltk.classify.SklearnClassifier(SVC(C=10, kernel='linear')).train(train_feats)
    print("  Accuracy: %f" % nltk.classify.accuracy(svm_classifier, test_feats))

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    filereader = FileReader()

    #print("##### Processing UD parsed sentences")
    #ud_dict, review_sentences = parse_ud(args[0])

    print('##### Reading in review data')
    review_dict, data_dict = filereader.parse_review_data(args[1])

    with open('urls.txt', 'r', encoding='utf-8') as url_file:
        urls = []
        url_lines = url_file.readlines()
        for url in url_lines:
            urls.append(url.rstrip())

    with open('odbrdata.txt', 'r', encoding='utf-8') as metadata_file:
        metadata_dict = {}
        metadata = csv.DictReader(metadata_file, delimiter='\t')
        for row in metadata:
            metadata_dict[row['url']] = {'accountid': row['accountid'], 'date': row['date'], 'rating': row['rating'], 'bookid': row['bookid'], 'nur': row['nur'], 'isbn': row['isbn'], 'author': row['author'], 'title': row['title']}

    #print('##### Generating word embeddings')
    #we_model = generate_word_embeddings()

    #print('##### Generating feature lexicon')
    #feature_lexicon = generate_feature_lexicon(data_dict, ud_dict, we_model)
    #for key, value in feature_lexicon.items():
    #    print(key, value)
    lexicon = {'1': {'geloofwaardigheid', 'ordeloosheid', 'technieken', 'vermengen', 'vermenging', 'plotlijn', 'vervlechten', 'op papier zetten', 'manier', 'aktie', 'benoemen', 'vocabulaire', 'hoeveelheid', 'intermezzo', 'toonzetting', 'geheel', 'bladvulling', 'penvoering', 'tijdsaanduiding', 'intelligentie', 'geneuzel', 'taalgebruik', 'betekenis', 'beschrijven', 'ruimtebeschrijvingen', 'orde', 'noteren', 'schrijfstijl', 'zinsnedes', 'onderbreken', 'hoofdstukken', 'situatiebeschrijvingen', 'ondertoon', 'smakelijk', 'materiaal', 'overgangen', 'materie', 'stijl', 'verbeelding', 'vaststelling', 'snelheid', 'vertaling', 'ingewikkelder', 'woorden/ontbrekende', 'zijl', 'uitdrukkingen', 'verhelderend', 'taalfouten', 'lectuur', 'verknopen', 'beeldspraak', 'verbinden', 'spanningsbogen', 'hoofdlijnen', 'gedachtegang', 'toelichting', 'krachttermen', 'woorden', 'schrijftechnisch', 'schrijven', 'zijlijntjes', 'fragmentarisch', 'vooruitwijzingen', 'omschrijvingen', 'structuur', 'woordkeus', 'toegankelijk', 'formuleren', 'uitwerking', 'woordenstroom', 'lijnen', "'rond' verhaal", 'verteltrant', "alinea's", 'woordgebruik', 'taalbeheersing', 'schrijfwijze', 'hoofdzakelijk', 'nadrukkelijkheid', 'ontwikkelingen', 'voetnoten', 'taalverzorging', 'woordspelingen', 'vernieuwingen', 'tekeningen', 'in elkaar zetten', 'compositie', 'toon', 'onsamenhangend', 'zinnen', 'taal', 'vertellen', 'eigenschap', 'elementen', 'verhelderen', 'verweving', 'uitweidingen', 'helderheid', 'woordkeuze', 'gebruiksvoorwerpen', 'onderbouwen', 'formuleringen', 'ogenschijnlijk', 'woordspeling', 'zinnetjes', 'voortgang', 'verhaallijn', 'weergeven', 'aaneenschakeling', 'patronen', 'sprongen in ruimte of tijd', 'perspectiefwisselingen', 'complimenten', 'managementsamenvatting', 'pretentieus', 'onoverzichtelijk', 'componeren', 'logica', 'teksten', 'vastigheid', 'opbouw', 'verhaallijnen', 'flash-backs', 'vergelijkingen', 'verteller'},
               '2': {'plooi', 'thema', 'episoden', 'hoofdthema', 'verhaal-idee', 'plotwending', 'verrassing', 'idee', 'thema’s', 'verrassend', 'climax', 'ontknoping', 'verhaallijn', 'onderwerp', 'onderwerpen', 'spanningsbogen', 'thematiek', 'geheel', "thema's", 'complot', 'spanning', 'plottwist', 'verhaal', 'onknoping', 'verhaaldraad', 'inzicht', 'einde', 'probleem', 'plot', 'handelingen', 'voorvallen', 'gedachte', 'cliffhanger', 'historie', 'verhaallijnen', 'misdaadverhaal', 'romanthese', 'visie', 'slot', 'problematiek', 'plotwendingen'},
               '3': {'verhalen', 'personeelsleden', 'mensen', 'situaties', 'vrouwen', 'krachten', 'hughes', 'ideaalbeelden', 'vlaanderen', 'mannen', 'gesprekken', 'type', 'karakter', 'trekjes', 'romanpersonage', 'edelheer', 'karikatuur', 'planten', 'bijrol', 'volgelingen', 'buren', 'redelijk', 'slechterik', 'hoofdrolspelers', 'gedachten', 'wijdelingen', 'emoties', 'hoofdpersonages', 'agnes', 'persoonlijkheid', 'hoofdrol', 'hoofdpersonen', 'vuisten', 'figuur', 'dagboekfragmenten', 'tv-persoonlijkheid', 'gedachtes', 'verhelderend', 'dantes', 'dialoog', 'vertakking', 'gebeurtenissen', 'persoonlijk', 'tegenspeler', 'hoofdfiguren', 'rustmomenten', 'spreken', 'zonen', 'rollen', 'dialogen', 'sujet', 'peeters', 'vaart', 'gevoelens', 'gesprek', 'kantelen', 'tatoeages', 'verdenkingen', 'agenten', 'typetjes', 'uiterlijk', 'dorpsgenoten', 'inzichten', 'eigenschappen', 'personage', 'uitwerpselen', 'medeleerlingen', 'sollen', 'huisgenoten', 'karel', 'dingen', 'beiden', 'personages', 'levensecht', 'held', 'karakters', 'boekpersonage', 'inhuren', 'feiten', 'subtiliteit', 'gasten', 'acties', 'woordenwisseling', 'mens', 'relaties', 'vertellen', 'redenen', 'verdachten', 'festiviteiten', 'feitelijk', 'personality', 'bijrollen', 'hoofdpersoon', 'hoofdpersonage', 'uitspraak', 'ontwikkeling', 'helderziend', 'woordenschat', 'protagonisten', 'dames', 'verhaallijn', 'kirsten', 'zussen', 'patiënten', 'bespiegelingen', 'speurders', 'geloofwaardig', 'quinten', 'aannames', 'familieleden'},
               '4': {'papier', 'illustratie', 'afbeelding', 'titel', 'uitgever', 'verschijnen', 'foto', 'presenteren', 'flaptekst'},
               '5': {'hoofdstuk', 'boek', 'deel', 'verhaal', 'vertelling', 'roman', 'werk', 'slot', 'r'}}

    feature_sets = generate_featuresets(lexicon, data_dict, urls, metadata_dict)
    train_feats, test_feats = split_feats(feature_sets)
    classification(train_feats, test_feats)

if __name__ == "__main__":
    main()