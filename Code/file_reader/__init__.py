from os.path import normpath # to read files
import conllu
import json

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
