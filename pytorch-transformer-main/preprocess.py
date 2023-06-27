# import csv

# f = open('./data/ko-en-translation.csv','r')
# csv_reader = csv.reader(f)
##
# for line in csv_reader:
#   print(line)

import csv

def load_csv(file_path):
    print(f'Load Data | file path: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        lines = []
        for line in csv_reader:
            line[0] = line[0].replace(';', '')
            lines.append(line)
    print(f'Load Complete | file path: {file_path}')
    return lines

if __name__ == "__main__":
    path = './data/ko-en-translation.csv'
    lines = load_csv(path)
    for line in lines:
        print(line)
