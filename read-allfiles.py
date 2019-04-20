from os import walk
import csv

f = []
f_pos = []
path = "C:\\Users\\asarwat\\Downloads\\imdb-movie-reviews-dataset\\aclImdb\\test\\neg"
pathPos = "C:\\Users\\asarwat\\Downloads\\imdb-movie-reviews-dataset\\aclImdb\\test\\pos"

for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(pathPos):
    f_pos.extend(filenames)
    break

dataNeg = []
dataPos = []

for filename in f_pos:
    with open(pathPos + "\\" + filename, encoding="utf8") as infile:
        text = infile.read()
        text = text.replace("\n", " ")
        dataPos.append(text)

for filename in f:
    with open(path + "\\" + filename, encoding="utf8") as infile:
        text = infile.read()
        text = text.replace("\n", " ")
        dataNeg.append(text)

with open("../dm_files/imdb_data_3K.csv", "w", encoding="utf8") as csvFile:
    writerCSV = csv.writer(csvFile, delimiter=',')
    writerCSV.writerow(['sentiment', 'text'])
    for i in range(0, 1500):
        writerCSV.writerow([1, dataPos[i]])
    for k in range(0, 1500):
        writerCSV.writerow([0, dataNeg[k]])

print('2')
