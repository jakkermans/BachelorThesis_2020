import sys
from __init__ import FileReader
from nltk.metrics import masi_distance
from nltk.metrics.agreement import AnnotationTask
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_iaa(data_dict):
    i = 0
    data = []
    for key, value in data_dict.items():
        i += 1
        data.append(('Annotator1', i, frozenset((value['label1'], value['label1_2']))))
        data.append(('Annotator2', i, frozenset((value['label2'], value['label2_2']))))

    print(data)
    t = AnnotationTask(data= data, distance=masi_distance)
    print(t.avg_Ao())

def calculate_iaa_label(number, data_dict):
    data = []
    y_true = []
    y_pred = []
    i = 0

    if number == 1:
        for key, value in data_dict.items():
            i += 1
            if value['label1'] in ['', ' ']:
                data.append(('Annotator1', str(i), '0'))
                y_pred.append('0')
            else:
                data.append(('Annotator1', str(i), value['label1']))
                y_pred.append(value['label1'])

            if value['label1_2'] in ['', ' ']:
                data.append(('Annotator2', str(i), '0'))
                y_true.append('0')
            else:
                data.append(('Annotator2', str(i), value['label1_2']))
                y_true.append(value['label1_2'])

        t = AnnotationTask(data)
        print('Cohen\'s Kappa for Label {}: {}'.format(number, t.pi()))

        matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(matrix, display_labels=["0", "1", "1+", "1-", "2", "2+", "2-", "3", "3+", "3-", "4", "4+", "4-", "5", "5+", "5-"])
        disp = disp.plot(include_values=True, values_format="d")

        fig = plt.gcf()
        fig.set_size_inches(6.5, 6.5)
        plt.xlabel('Annotator2')
        plt.ylabel('Annotator1')
        plt.title('Agreement Label 1')

        plt.show()
        plt.savefig('agreement_label1')

    else:
        for key, value in data_dict.items():
            i += 1
            if value['label2'] in ['', ' ']:
                data.append(('Annotator1', str(i), '0'))
                y_pred.append('0')
            else:
                data.append(('Annotator1', str(i), value['label2']))
                y_pred.append(value['label2'])

            if value['label2_2'] in ['', ' ']:
                data.append(('Annotator2', str(i), '0'))
                y_true.append('0')
            else:
                data.append(('Annotator2', str(i), value['label2_2']))
                y_true.append(value['label2_2'])

        t = AnnotationTask(data)
        print('Cohen\'s Kappa for Label {}: {}'.format(number, t.pi()))

        matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(matrix,
                                      display_labels=["0", "1", "1+", "1-", "2", "2+", "2-", "3", "3+", "3-", "4", "4+",
                                                      "4-", "5", "5+", "5-"])
        disp = disp.plot(include_values=True, values_format="d")

        fig = plt.gcf()
        fig.set_size_inches(6.5, 6.5)
        plt.xlabel('Annotator2')
        plt.ylabel('Annotator1')
        plt.title('Agreement Label 2')

        plt.show()
        plt.savefig('agreement_label2')

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    filereader = FileReader()

    print('##### Reading in review data')
    review_list, data_list = filereader.parse_review_data(args[0])
    print(len(data_list))
    data_dict = {}

    for item in review_list:
        for key, value in item.items():
            data_dict[key] = value

    print("##### Calculating Inter-Annotator Agreement")
    calculate_iaa(data_dict)

    calculate_iaa_label(1, data_dict)
    calculate_iaa_label(2, data_dict)

if __name__ == "__main__":
    main()