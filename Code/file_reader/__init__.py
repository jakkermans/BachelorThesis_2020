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
            self.review_list = []
            self.training_list = []
            self.review_data = data_file.readlines()
            for row in self.review_data[1:]:
                row = row.rstrip().split('\t')
                self.token_list = []
                self.review_sentence = row[4].split()
                for token in self.review_sentence:
                    if token not in self.punct_list:
                        self.token_list.append(token.lower())
                self.review_list.append({row[6]: {'label1': row[0], 'label2': row[1],
                                              'label1_2': row[2], 'label2_2': row[3],
                                              'sentence': self.token_list, 'filename': row[5]}})
                if (row[0] != '' and row[1] == '') or (row[0] != '' and row[1] == ''):
                    self.training_list.append({row[6]: {'label1': row[0], 'label2': row[1],
                                                      'label1_2': row[2], 'label2_2': row[3],
                                                      'sentence': self.token_list, 'filename': row[5]}})

        return self.review_list, self.training_list
