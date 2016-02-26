# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 11:00:45 2016

Configuration for conversation analysis.

@author: anshmania
"""
import imp
import os
#import shutil
import urllib
import zipfile
from pymongo import MongoClient

server='localhost'
port=27017
database_name='test'
collection_name='chatSessions'

# MongoDB connection
mongo_client=MongoClient(server,port)

#Setting the paths
nltk_path=imp.find_module('nltk')[1]
nltk_sentiment_path = nltk_path + '\\sentiment'

# Stanford files setup
stanford_pos_tagger='http://nlp.stanford.edu/software/stanford-postagger-full-2014-08-27.zip'
stanford_ner_tagger='http://nlp.stanford.edu/software/stanford-ner-2015-04-20.zip'
stanford_pos_path = nltk_sentiment_path+'\\stanford-postagger-full-2014-08-27.zip'
stanford_ner_path = nltk_sentiment_path+'\\stanford-ner-2015-04-20.zip'

if os.path.exists(nltk_sentiment_path+'\\stanford-postagger-full-2014-08-27\\stanford-postagger.jar')==True:
    pass
else:
    stanford_pos_file=urllib.urlretrieve(stanford_pos_tagger, stanford_pos_path)
    stanford_ner_file=urllib.urlretrieve(stanford_ner_tagger, stanford_ner_path)
    #Extract the file to nltk sentiment directory
    zipfile.ZipFile.extractall(zipfile.ZipFile(stanford_pos_path),nltk_sentiment_path)
    zipfile.ZipFile.extractall(zipfile.ZipFile(stanford_ner_path),nltk_sentiment_path)


# Set file paths to NER and POS jar files from the working directory
stanford_pos_jar_file_location = nltk_sentiment_path+'\\stanford-postagger-full-2014-08-27\\stanford-postagger.jar'
stanford_pos_tagger_file_location = nltk_sentiment_path+'\\stanford-postagger-full-2014-08-27\\models\\english-bidirectional-distsim.tagger'
stanford_ner_jar_file_location = nltk_sentiment_path+'\\stanford-ner-2015-04-20\\stanford-ner.jar'
stanford_ner7_tagger_file_location = nltk_sentiment_path+'\\stanford-ner-2015-04-20\\classifiers\\english.muc.7class.distsim.crf.ser.gz'
stanford_ner4_tagger_file_location = nltk_sentiment_path+'\\stanford-ner-2015-04-20\\classifiers\\english.conll.4class.distsim.crf.ser.gz'

##Clean zip files
#os.remove(stanford_pos_path)
#os.remove(stanford_ner_path)
# Retrieve Qumram Lexicon files.
url_qumram_lexicon = 'https://github.com/anshmania/Python-codes/blob/master/qumram_lexicon.txt'
url_vader_lexicon = 'https://github.com/anshmania/Python-codes/blob/master/vader_lexicon.txt'
qumram_lexicon_path=nltk_sentiment_path+'\\qumram_lexicon.txt'
vader_lexicon_path=nltk_sentiment_path+'\\vader_lexicon.txt'

if os.path.exists(nltk_sentiment_path+'\\qumram_lexicon.txt')==True:
    pass
else:
    files_zipped=urllib.urlretrieve('https://github.com/anshmania/Python-codes/blob/master/zipped_lexicon.zip?raw=true',nltk_sentiment_path+'\\zipped_lexicons.zip')
    zipfile.ZipFile.extractall(zipfile.ZipFile(files_zipped[0]),nltk_sentiment_path)


# Feature controls for NLP
minimum_length_of_word_to_be_considerd_longword=9
minimum_frequency_of_words_to_be_considered_high_frequency_=2
minimum_frequency_of_words_to_be_considered_bi_trigrams=2
number_of_bigrams_to_be_extracted=3
number_of_trigrams_to_be_extracted=3

