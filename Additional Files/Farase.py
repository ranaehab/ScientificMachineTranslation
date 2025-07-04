from farasa.pos import FarasaPOSTagger
from farasa.ner import FarasaNamedEntityRecognizer
from farasa.diacratizer import FarasaDiacritizer
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer
import csv


f1 = open('SourceFile.txt', 'r',encoding='utf8')
for line in f1:
    segmenter = FarasaSegmenter()
    segmented = segmenter.segment(line)
    f2 = open('OutputFile.txt', 'a',encoding='utf8')
    f2.write(segmented)
    f2.write('\n')
    f2.close()
