from os.path import normpath # to read files
import conllu
import json
import csv

class FileReader():
    def parse_conllu_data(self, parse_files):
        self.path =  normpath("D:\Informatiekunde Jaar 3\Bachelor Scriptie\\parses-uncompressed")
        self.exclude = ['output_part00300.conllu']
        with open('processed_output_v2.txt', 'w', encoding='utf-8') as output_file:
            for file in parse_files:
                if file not in self.exclude:
                    print(file)
                    data = open(self.path + '\\' + file, 'r', encoding='utf-8')
                    for token_list in conllu.parse_incr(data):
                        token_data = []
                        for token in token_list:
                            token_data.append((token['form'], token['upostag']))
                        output_file.write("{}   {}\n".format(token_list.metadata['sent_id'], " ".join("(%s,%s)" % tup for tup in token_data)))

    def parse_review_data(self, review_file):
        with open(review_file, 'r', encoding='utf-8') as data_file:
            self.punct_list = ['.', ',', '?', ':', '(', ')', '!', "'", '`', '...', '``', "''", '"', "’", "”", "“", "’", "-",
                          ";", "‘", "="]
            self.review_dict = {}
            self.training_dict = {}
            self.review_data = csv.DictReader(data_file, delimiter='\t')
            for row in self.review_data:
                self.token_list = []
                self.review_sentence = row['sentence'].split()
                for token in self.review_sentence:
                    if token not in self.punct_list:
                        self.token_list.append(token.lower())
                self.review_dict[row['sentid']] = {'label1': row['label1'], 'label2': row['label2'],
                                              'label1_2': row['label1_2'], 'label2_2': row['label2_2'],
                                              'sentence': self.token_list, 'filename': row['filename'], 'fold': row['fold']}
                if (row['label1'] != '' and row['label2'] == '') or (row['label1'] != '' and row['label2'] == ''):
                    self.training_dict[row['sentid']] = {'label1': row['label1'], 'label2': row['label2'],
                                              'label1_2': row['label1_2'], 'label2_2': row['label2_2'],
                                              'sentence': self.token_list, 'filename': row['filename'], 'fold': int(row['fold'])}

        return self.review_dict, self.training_dict
