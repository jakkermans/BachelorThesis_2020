import sys
from __init__ import FileReader
from gensim.models import FastText
import csv
import re
from random import shuffle
from sklearn.svm import SVC, LinearSVC
import nltk.classify
import collections, itertools
from nltk.metrics import precision, recall
from sklearn.metrics import confusion_matrix, classification_report


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
                    if pos_data[token] == 'propn':
                        feature_lexicon['3'].add(token)
                    elif pos_data[token] in ['noun', 'adj', 'verb', 'propn', 'adp']:
                        if token not in feature_lexicon[review_data['label1'][0]]:
                            for aspect in aspect_dict[review_data['label1'][0]]:
                                similarity_scores.append(we_model.wv.similarity(token, aspect))
                                if max(similarity_scores) > 0.2:
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

def generate_featurelexicon_memberbased(data_dict, ud_dict, we_model):
    feature_lexicon = {
        '1': {'stijl', 'taal', 'toon', 'ondertoon', 'penvoering', 'zinnetjes', 'zinnen', 'toonzetting', 'woorden',
              'woordspelingen', 'woordenstroom', 'formuleringen', 'taalgebruik', 'vocabulaire', 'verteltrant',
              'schrijfwijze', 'taalbeheersing', 'woordkeus', 'schrijven', 'beschrijven', 'weergeven', 'vertellen',
              'benoemen', 'formuleren', 'op papier zetten', 'noteren', 'structuur', 'orde', 'ordeloosheid',
              'sprongen in ruimte of tijd', 'vermenging', 'fragmentarisch', 'hoofdstukken', 'alinea\'s', 'lijnen',
              'hoofdlijnen', 'zijlijntjes', 'vooruitwijzingen', 'flash-backs', 'patronen', 'perspectiefwisselingen',
              '\'rond\' verhaal', 'compositie', 'vervlechten', 'in elkaar zetten', 'componeren', 'verknopen',
              'onderbreken', 'verbinden', 'vermengen'},
        '2': {'plot', 'verhaal', 'verhaaldraad', 'slot', 'ontknoping', 'historie', 'handelingen', 'episoden',
              'voorvallen', 'thema', 'idee', 'probleem', 'problematiek', 'gedachte', 'romanthese', 'visie', 'inzicht'},
        '3': {'personages', 'figuur', 'hoofdpersoon', 'karakter', 'mens', 'tegenspeler', 'hoofdrol', 'bijrol',
              'persoonlijkheid', 'sujet', 'held', 'type', 'dialoog', 'gesprek', 'woordenwisseling', 'uitspraak',
              'vertellen', 'spreken'},
        '4': {'uitgever', 'titel', 'illustratie', 'afbeelding', 'foto', 'flaptekst', 'papier', 'verschijnen',
              'presenteren'},
        '5': {'boek', 'werk', 'roman', 'verhaal', 'vertelling', 'deel', 'hoofdstuk', 'slot'}}

    aspect_dict = {'1': ['stijl', 'structuur'],
                   '2': ['plot', 'thema'],
                   '3': ['karakters', 'dialoog'],
                   '4': 'verschijning',
                   '5': 'gehele werk'}

    member_dict = {}

    for key, value in aspect_dict.items():
        members = []
        for aspect in value:
            try:
                members += [member[0] for member in we_model.wv.most_similar(aspect)][:50]
            except IndexError:
                members += [member[0] for member in we_model.wv.most_similar(aspect)][:len([member[0] for member in we_model.wv.most_similar(aspect)])]
        member_dict[key] = set(members)

    for key, value in data_dict.items():
        try:
            review_data = value
            pos_data = ud_dict[key]
            for token in review_data['sentence']:
                similarity_scores = []
                try:
                    if pos_data[token] == 'propn':
                        feature_lexicon['3'].add(token)
                    elif pos_data[token] in ['noun', 'adj', 'verb', 'propn', 'adp']:
                        if token not in feature_lexicon[review_data['label1'][0]]:
                            if token in member_dict[review_data['label1'][0]]:
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
    train_feats = []
    dev_feats = []
    test_feats = []
    for key, value in data_dict.items():
        features = {}
        id_match = re.match('(\d+)_', key)
        for token in value['sentence']:
            if token in lexicon[value['label1'][0]]:
                features[token] = 1
            else:
                features[token] = 0
        review_url = urls[int(id_match.group(1))+1]
        try:
            nur_code = metadata_dict[review_url]['nur']
            features[nur_code] = 1
        except KeyError:
            features['None'] = 1

        if value['fold'] >= 1 and value['fold'] <= 8:
            train_feats.append((features, value['label1'][0]))
        elif value['fold'] == 9:
            dev_feats.append((features, value['label1'][0]))
        else:
            test_feats.append((features, value['label1'][0]))

    split = int(len(train_feats) * 0.8)

    return train_feats[:split], dev_feats, test_feats

def generate_bow_features(data_dict, urls, metadata_dict, ud_dict):
    train_feats = []
    dev_feats = []
    test_feats = []
    for key, value in data_dict.items():
        try:
            features = {}
            id_match = re.match('(\d+)_', key)
            pos_data = ud_dict[key]
            for word in value['sentence']:
                if pos_data[word] in ['noun', 'adj', 'verb', 'propn', 'adp']:
                    features[word] = 1
                else:
                    features[word] = 1
            review_url = urls[int(id_match.group(1)) + 1]
            try:
                nur_code = metadata_dict[review_url]['nur']
                features['nur'] = 1
            except KeyError:
                features['nur'] = 0
            if value['fold'] >= 1 and value['fold'] <= 8:
                train_feats.append((features, value['label1'][0]))
            elif value['fold'] == 9:
                dev_feats.append((features, value['label1'][0]))
            else:
                test_feats.append((features, value['label1'][0]))
        except KeyError:
            pass

    return train_feats, dev_feats, test_feats

def classification(train_feats):
    print("##### Classifying book reviews")
    svm_classifier = nltk.classify.SklearnClassifier(SVC(C=1, kernel='linear')).train(train_feats)
    return svm_classifier

def evaluation(svm_classifier, test_feats):
    print("##### Evaluating classification")
    p_value = r_value = 0
    print("  Accuracy: %f" % nltk.classify.accuracy(svm_classifier, test_feats))
    precision, recall = precision_recall(svm_classifier, test_feats)
    for (key, precision_value), (key2, recall_value) in zip(precision.items(), recall.items()):
        try:
            p_value += precision_value
        except TypeError:
            p_value += 0
        try:
            r_value += recall_value
        except TypeError:
            r_value += 0

    avg_precision = p_value/len(precision.keys())
    avg_recall = r_value/len(recall.keys())

    f_score = (2 * (avg_precision * avg_recall)) / (avg_precision + avg_recall)

    print("  Precision: %f" % avg_precision)
    print("  Recall: %f" % avg_recall)
    print("  F1-score: %f" % f_score)

    print("##### Metrics per category")
    categories = precision.keys()
    print(" |--------------------|--------------------|--------------------|--------------------|--------------------|")
    print(" |%-20s|%-20s|%-20s|%-20s|%-20s|" % ("category", "accuracy", "precision", "recall", "F-measure"))
    print(" |--------------------|--------------------|--------------------|--------------------|--------------------|")
    for category in categories:
        if precision[category] is None:
            print(" |%-20s|%-20s|%-20s|%-20s|%-20s|" % (category, recall[category], "NA", recall[category], "NA"))
        else:
            try:
                print(" |%-20s|%-20s|%-20s|%-20s|%-20s|" % (
                category, recall[category], precision[category], recall[category], (2 * (precision[category] * recall[category])) / (precision[category] + recall[category])))
            except ZeroDivisionError:
                print(" |%-20s|%-20s|%-20s|%-20s|%-20s|" % (
                    category, recall[category], precision[category], recall[category],
                    0))
    print(" |--------------------|--------------------|--------------------|--------------------|--------------------|")


def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    filereader = FileReader()

    print("##### Processing UD parsed sentences")
    ud_dict, review_sentences = parse_ud(args[0])

    print('##### Reading in review data')
    review_list, data_list = filereader.parse_review_data(args[1])
    data_dict = {}

    for item in data_list:
        for key, value in item.items():
            data_dict[key] = value

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

    print('##### Generating word embeddings')
    we_model = generate_word_embeddings()

    print('##### Generating feature lexicon')
    feature_lexicon = generate_feature_lexicon(data_dict, ud_dict, we_model)
    #feature_lexicon = generate_featurelexicon_memberbased(data_dict, ud_dict, we_model)
    #lexicon = {'1': {'geloofwaardigheid', 'ordeloosheid', 'technieken', 'vermengen', 'vermenging', 'plotlijn', 'vervlechten', 'op papier zetten', 'manier', 'aktie', 'benoemen', 'vocabulaire', 'hoeveelheid', 'intermezzo', 'toonzetting', 'geheel', 'bladvulling', 'penvoering', 'tijdsaanduiding', 'intelligentie', 'geneuzel', 'taalgebruik', 'betekenis', 'beschrijven', 'ruimtebeschrijvingen', 'orde', 'noteren', 'schrijfstijl', 'zinsnedes', 'onderbreken', 'hoofdstukken', 'situatiebeschrijvingen', 'ondertoon', 'smakelijk', 'materiaal', 'overgangen', 'materie', 'stijl', 'verbeelding', 'vaststelling', 'snelheid', 'vertaling', 'ingewikkelder', 'woorden/ontbrekende', 'zijl', 'uitdrukkingen', 'verhelderend', 'taalfouten', 'lectuur', 'verknopen', 'beeldspraak', 'verbinden', 'spanningsbogen', 'hoofdlijnen', 'gedachtegang', 'toelichting', 'krachttermen', 'woorden', 'schrijftechnisch', 'schrijven', 'zijlijntjes', 'fragmentarisch', 'vooruitwijzingen', 'omschrijvingen', 'structuur', 'woordkeus', 'toegankelijk', 'formuleren', 'uitwerking', 'woordenstroom', 'lijnen', "'rond' verhaal", 'verteltrant', "alinea's", 'woordgebruik', 'taalbeheersing', 'schrijfwijze', 'hoofdzakelijk', 'nadrukkelijkheid', 'ontwikkelingen', 'voetnoten', 'taalverzorging', 'woordspelingen', 'vernieuwingen', 'tekeningen', 'in elkaar zetten', 'compositie', 'toon', 'onsamenhangend', 'zinnen', 'taal', 'vertellen', 'eigenschap', 'elementen', 'verhelderen', 'verweving', 'uitweidingen', 'helderheid', 'woordkeuze', 'gebruiksvoorwerpen', 'onderbouwen', 'formuleringen', 'ogenschijnlijk', 'woordspeling', 'zinnetjes', 'voortgang', 'verhaallijn', 'weergeven', 'aaneenschakeling', 'patronen', 'sprongen in ruimte of tijd', 'perspectiefwisselingen', 'complimenten', 'managementsamenvatting', 'pretentieus', 'onoverzichtelijk', 'componeren', 'logica', 'teksten', 'vastigheid', 'opbouw', 'verhaallijnen', 'flash-backs', 'vergelijkingen', 'verteller'},
    #           '2': {'plooi', 'thema', 'episoden', 'hoofdthema', 'verhaal-idee', 'plotwending', 'verrassing', 'idee', 'thema’s', 'verrassend', 'climax', 'ontknoping', 'verhaallijn', 'onderwerp', 'onderwerpen', 'spanningsbogen', 'thematiek', 'geheel', "thema's", 'complot', 'spanning', 'plottwist', 'verhaal', 'onknoping', 'verhaaldraad', 'inzicht', 'einde', 'probleem', 'plot', 'handelingen', 'voorvallen', 'gedachte', 'cliffhanger', 'historie', 'verhaallijnen', 'misdaadverhaal', 'romanthese', 'visie', 'slot', 'problematiek', 'plotwendingen'},
    #           '3': {'verhalen', 'personeelsleden', 'mensen', 'situaties', 'vrouwen', 'krachten', 'hughes', 'ideaalbeelden', 'vlaanderen', 'mannen', 'gesprekken', 'type', 'karakter', 'trekjes', 'romanpersonage', 'edelheer', 'karikatuur', 'planten', 'bijrol', 'volgelingen', 'buren', 'redelijk', 'slechterik', 'hoofdrolspelers', 'gedachten', 'wijdelingen', 'emoties', 'hoofdpersonages', 'agnes', 'persoonlijkheid', 'hoofdrol', 'hoofdpersonen', 'vuisten', 'figuur', 'dagboekfragmenten', 'tv-persoonlijkheid', 'gedachtes', 'verhelderend', 'dantes', 'dialoog', 'vertakking', 'gebeurtenissen', 'persoonlijk', 'tegenspeler', 'hoofdfiguren', 'rustmomenten', 'spreken', 'zonen', 'rollen', 'dialogen', 'sujet', 'peeters', 'vaart', 'gevoelens', 'gesprek', 'kantelen', 'tatoeages', 'verdenkingen', 'agenten', 'typetjes', 'uiterlijk', 'dorpsgenoten', 'inzichten', 'eigenschappen', 'personage', 'uitwerpselen', 'medeleerlingen', 'sollen', 'huisgenoten', 'karel', 'dingen', 'beiden', 'personages', 'levensecht', 'held', 'karakters', 'boekpersonage', 'inhuren', 'feiten', 'subtiliteit', 'gasten', 'acties', 'woordenwisseling', 'mens', 'relaties', 'vertellen', 'redenen', 'verdachten', 'festiviteiten', 'feitelijk', 'personality', 'bijrollen', 'hoofdpersoon', 'hoofdpersonage', 'uitspraak', 'ontwikkeling', 'helderziend', 'woordenschat', 'protagonisten', 'dames', 'verhaallijn', 'kirsten', 'zussen', 'patiënten', 'bespiegelingen', 'speurders', 'geloofwaardig', 'quinten', 'aannames', 'familieleden'},
    #           '4': {'papier', 'illustratie', 'afbeelding', 'titel', 'uitgever', 'verschijnen', 'foto', 'presenteren', 'flaptekst'},
    #           '5': {'hoofdstuk', 'boek', 'deel', 'verhaal', 'vertelling', 'roman', 'werk', 'slot', 'r'}}

    train_feats, dev_feats, test_feats = generate_featuresets(feature_lexicon, data_dict, urls, metadata_dict)
    final_train_feats = train_feats + dev_feats
    #train_feats, dev_feats, test_feats = generate_bow_features(data_dict, urls, metadata_dict, ud_dict)
    model = classification(train_feats)

    y_true = [feat[1] for feat in dev_feats]
    testing_feats = [feat[0] for feat in dev_feats]

    print(confusion_matrix(y_true, model.classify_many(testing_feats)))
    print(classification_report(y_true, model.classify_many(testing_feats), zero_division=False))
    #evaluation(model, test_feats)
    #counts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    #for item in review_list:
    #    for key, value in item.items():
    #        try:
    #            counts[value['label1'][0]] += 1
    #        except IndexError:
    #            pass

    #print(counts)

if __name__ == "__main__":
    main()