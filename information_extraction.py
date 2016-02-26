# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:22:09 2016

@author: anshmania
"""
from collections import defaultdict
import nltk
from pymongo import MongoClient
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
import timex
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures ,BigramAssocMeasures
from criticality_compliance import sentiment as senti
from nltk.corpus import stopwords
import re
import configNLP


st = StanfordPOSTagger(configNLP.stanford_pos_tagger_file_location,configNLP.stanford_pos_jar_file_location)
ne7=StanfordNERTagger(configNLP.stanford_ner7_tagger_file_location,configNLP.stanford_ner_jar_file_location)
ne4=StanfordNERTagger(configNLP.stanford_ner4_tagger_file_location,configNLP.stanford_ner_jar_file_location)

#st=StanfordPOSTagger('C:\\Users\\anshmania\\Anaconda2\\Lib\\site-packages\\nltk\\sentiment\\stanford-postagger-full-2014-08-27\\models\\english-bidirectional-distsim.tagger','C:\\Users\\anshmania\\Anaconda2\\Lib\\site-packages\\nltk\\sentiment\\stanford-postagger-full-2014-08-27\\stanford-postagger.jar')

client=MongoClient()

# returns string
def textExtractor(conversationId):
    conversation= client.test.chatSessions.find({'conversationId':conversationId})  #cursor object
    for data in conversation:
        text=data['text']
    return ' '.join(text)

#########################################################################
 #using nltk english stopwords
# Remove stop words and return list of tokenized words
def removeStops(conversationId):
    english_stops=set(stopwords.words('english'))
    try:
        text=textExtractor(conversationId)
        word_list= nltk.word_tokenize(text)
        stop_free_text=[word for word in word_list if word not in english_stops]
            #return ''.join(stop_free_text)
        return stop_free_text
    except:
        None
#####################################################################
################# Information Extraction ############################
#####################################################################

# Long words extraction
def extractLongWords(conversationId):
    try:
        longWords=defaultdict(list)
        text=removeStops(conversationId)
        text=nltk.word_tokenize(' '.join(text))
        longWords=[word for word in text if len(word)>=9]
        return longWords
    except:
        None
# High frequency words of conversation
def highFrequencyWords(conversationId):
    try:
        highFreq=defaultdict(list)
        text=removeStops(conversationId)
        textFreq=nltk.FreqDist(nltk.word_tokenize(' '.join(text)))
        highFreq=[word for word in textFreq.keys() if textFreq[word]>=2 and len(word) >4]
        return highFreq
    except:
        None

# Chunk the noun phrases
def chunkedNNP(conversationId):
    #chunkNNP=defaultdict(list)
    text=textExtractor(conversationId)
    token_text=nltk.word_tokenize(text)
    tagged_text=st.tag(token_text)
    grammar='''NP: {<DT|PP\$>?<JJ>*<NN>}
                {<NNP>+}
                {<NN>+}'''
        #Loca="LOCATION: {<IN>?<JJ>*<NN>}"
    parser=nltk.RegexpParser(grammar)
        #loc_parser=nltk.RegexpParser(Loca)
    tree = parser.parse(tagged_text)
    return tree

def chunker(conversationId):
        text=textExtractor(conversationId)
        token_text=nltk.word_tokenize(text)
        tagged_text=nltk.pos_tag(token_text)
        return nltk.ne_chunk(tagged_text)

def stanfordChunker(conversationId):
        text=textExtractor(conversationId)
        token_text=nltk.word_tokenize(text)
        #tagged_text=nltk.pos_tag(token_text)
        return ne7.tag(token_text)

# Extract named entities from unstructured text
def chunkedEntities(conversationId):
    labelss=defaultdict(list)
    for item in chunkedNNP(conversationId):
        if isinstance(item,nltk.tree.Tree):
            for tuples in item.leaves():
                labelss.setdefault(item.label(),[]).append(tuples[0])
    return labelss

# Extract named entities from unstructured text

def namedEntitiesNEChunk(conversationId):
    labelss=defaultdict(list)
    for item in chunker(conversationId):
        if isinstance(item,nltk.tree.Tree):
            for tuples in item.leaves():
                labelss.setdefault(item.label(),[]).append(tuples[0])
    return labelss


def namedEntitiesStanfordChunk(conversationId):
    labelsStanford=defaultdict(list)
    for item in stanfordChunker(conversationId):
        word,tag = item
        if tag != 'O' and len(word)>4:
            labelsStanford.setdefault(tag,[]).append(word)
        else:
            None
    return labelsStanford



# Date information.
def dateEntities(conversationId):
    try:
        text=textExtractor(conversationId)
        tag_data=timex.tag(text)
        return tag_data
    except:
        None

# Money information.
def currencyEntities(conversationId):
    try:
        currency=["Dollars","Dollar","Euros","Euro","Pounds","Pound","GBP","INR","$"]
        text=textExtractor(conversationId)
        reg = "\d+.\w+"
        regExp = re.compile(reg)
        found=regExp.findall(text)
        for item in found:
            item=item.split()
            for word in item:
                if word in currency:
                    return ' '.join(item)
                else:
                    None
    except:
        None

# Time information.
def timeEntities(conversationId):
    try:
        time=["PM","AM","P.M.","A.M.","pm","am","o\'clock"]
        text=textExtractor(conversationId)
        reg1 = "\d+.\w+"
        regExp1 = re.compile(reg1)
        found=regExp1.findall(text)
        for item in found:
            item=item.split()
            for word in item:
                if word in time:
                    return ' '.join(item)
                else:
                    None
    except:
        None


#########################################################################
# Collocations should be found for the whole text, once they have been collected for every individual conversation
def biCollocationFinder(conversationId):
    try:
        filter_stops=lambda w: len(w)<3
        bcf = BigramCollocationFinder.from_words(removeStops(conversationId))
        bcf.apply_word_filter(filter_stops)
        bcf.apply_freq_filter(2) #filter to find collocations appearing atleast twice
        bi_likelihood=bcf.nbest(BigramAssocMeasures.likelihood_ratio, 3)
    #bi_chi=bcf.nbest(BigramAssocMeasures.chi_sq, 3)
    #bi_fish=bcf.nbest(BigramAssocMeasures.fisher, 3)
        return bi_likelihood
    except:
        None
# trigram collocations
def triCollocationFinder(conversationId):
    try:
        filter_stops=lambda w: len(w)<3
        tcf = TrigramCollocationFinder.from_words(removeStops(conversationId))
        tcf.apply_word_filter(filter_stops)
        tcf.apply_freq_filter(2)
        triGrams=tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 3)
        return triGrams#,bi_chi,bi_fish
    except:
        None
#########################################################################
# calculating the sentiment score of collocations from a text
def biCollocationScore(conversationId):
    try:
        bi_list=[]
        bi=biCollocationFinder(conversationId)
        [bi_list.append(senti(word)['non-compliance']) for item in bi for word in item]
        sum_b=sum(bi_list)                    # a way to normalize these scores???
        return sum_b
    except:
        None
def triCollocationScore(conversationId):
    try:
        tri_list=[]
        tri=triCollocationFinder(conversationId)
        [tri_list.append(senti(word)['non-compliance']) for item in tri for word in item]
        sum_t=sum(tri_list)
        return sum_t
    except:
        None

#########################################################################
# give the abstract of the conversation.
def abstraction(conversationId):
    highFreqWords=highFrequencyWords(conversationId)
    longWordsInConversation=extractLongWords(conversationId)
    abstraction=[]
    for word in highFreqWords:
        if word in longWordsInConversation:
            abstraction.append(word)
    return abstraction



#########################################################################

def infoExtractor(conversationId):
    info={}
    info['PEOPLE']=set(namedEntitiesStanfordChunk(conversationId)['PERSON'])
    info['ORGANISATION']=set(namedEntitiesStanfordChunk(conversationId)['ORGANIZATION'])
    info['LOCATION']=set(namedEntitiesStanfordChunk(conversationId)['LOCATION'])
    info['HIGH FREQUENCY TERMS']=set(highFrequencyWords(conversationId))
    info['LONG WORDS']=set(extractLongWords(conversationId))
    info['DATE']=dateEntities(conversationId)
    info['TIME']=timeEntities(conversationId)
    info['MONEY']=currencyEntities(conversationId)
    info['BI_GRAMS']=biCollocationFinder(conversationId)
    info['TRI_GRAMS']=triCollocationFinder(conversationId)
    info['ABSTRACTION']=abstraction(conversationId)
    return info