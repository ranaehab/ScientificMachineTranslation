from farasa.pos import FarasaPOSTagger
from farasa.ner import FarasaNamedEntityRecognizer
from farasa.diacratizer import FarasaDiacritizer
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer
import csv

#source: https://r12a.github.io/scripts/tutorial/summaries/arabic
sample =\
'''
دراسه دور انزيم التيلوميراز العكسي في تنظيم التعبير الجيني لبعض الجينات.
'''
f1 = open('D:/PHD Papers/scientific translation/Technical/Farsa/LastDataset/New Text Document.txt', 'r',encoding='utf8')
for line in f1:
    segmenter = FarasaSegmenter()
    segmented = segmenter.segment(line)
    f2 = open('D:/PHD Papers/scientific translation/Technical/Farsa/LastDataset/TestFarsa.txt', 'a',encoding='utf8')
    f2.write(segmented)
    f2.write('\n')
    f2.close()



