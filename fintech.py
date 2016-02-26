# -*- coding: iso-8859-15 -*-
"""
Created on Fri Jan 15 10:46:47 2016

@author: anshmania
"""

import nltk
#from nltk.book import *
from collections import defaultdict
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import criticality_compliance as cnc
from information_extraction import removeStops
#import matplotlib.pyplot as plt
from scipy.special import expit
import information_extraction as ie
import configNLP
#import timex



# MongoDC client
client=configNLP.mongo_client
db=configNLP.database_name
collection=configNLP.collection_name


keyWords=['sale','username','password','usr','pass','buy','price','try', 'policy','would', 'new','will','like','purchase','insurance','id','under']
keyWords1=['haywire','problematic','problem','doesnt work','work','which']

#########################################################################
################# Features for scoring conversations ####################
#########################################################################

# Frequency distribution of keywords in conversations
def keywordFreqAnalyser(conversationId):
    freqData=defaultdict(list)
    conversation= client.db.collection.find({'conversationId':conversationId})
    for data in conversation:
        freqDist=nltk.FreqDist(data['text'])
        for word  in freqDist.iterkeys():
            if word in keyWords:
                freqData.setdefault(data['conversationId'],[]).append(freqDist[word])
    return sum(freqData[conversationId])



#########################################################################
# finding the degree of non-compliance in a conversation
def nonCompliance(conversationId):
   text=' '.join(removeStops(conversationId)) #use when text is stored as string in database
   #text=removeStops(conversationId)          #use when text is stored as list
   sentiment=cnc.sentiment(text)
   #sentiment=cnc.sentiment(text)
   return sentiment['non-compliance']

#########################################################################
## Take the long words and high frequency words of the text into account
#f = configNLP.qumram_lexicon_file[0]
WORD_VALENCE_DICT = cnc.WORD_VALENCE_DICT

def valenceScore(keyword):
    try: WORD_VALENCE_DICT[keyword]
    except KeyError: return 0

def longWordScore(conversationId):
    long_word_score=0
    for word in ie.extractLongWords(conversationId):
        if word in WORD_VALENCE_DICT.keys():
            try:
                long_word_score += valenceScore(word)
            except: return 0
        else:
            return 0

def highFreqScore(conversationId):
    i=0
    j=0
    for word in ie.highFrequencyWords(conversationId):
        if word in keyWords:
            i += 1
        elif word in keyWords1:
            j += 1
        else:
            pass
    return (i,j)


#########################################################################
######################### Analysis ######################################
#########################################################################

# putting all the features together and creating a criticality index
def featureExtractor(conversationId):
    keywordScore = keywordFreqAnalyser(conversationId)
    sum_feature= keywordScore+nonCompliance(conversationId)\
    +ie.biCollocationScore(conversationId)+ie.triCollocationScore(conversationId)\
    +abs(longWordScore(conversationId))\
    +sum(highFreqScore(conversationId))
#    return logistic._pdf(sum_feature)
    if sum_feature<0.5:
        sum_feature -= 1
        return expit(sum_feature)
    else:
        return expit(sum_feature)

#########################################################################
# Sentiment Analysis of chat sessions
def sentimentAnalyser(conversationId):
    sentences=[]
    sentimentData=defaultdict(list)
    sid=SentimentIntensityAnalyzer(lexicon_file=configNLP.vader_lexicon_path)
    text=ie.textExtractor(conversationId)
    lines_list= tokenize.sent_tokenize(text)
    sentences.append(lines_list)
    for i,sentence in enumerate(sentences):
        sentence=' '.join(sentence)
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            sentimentData.setdefault(conversationId,[]).append({k:ss[k]})
    sentences=[]
    return sentimentData[conversationId]

def returnValues(conversationId):
    conversation_analysis={}
    conversation_analysis['SENTIMENT']=sentimentAnalyser(conversationId)
    conversation_analysis['CRITICALITY']=featureExtractor(conversationId)
    conversation_analysis['INFORMATION']=ie.infoExtractor(conversationId)
    return conversation_analysis

######################################################################
## Declare some variables and Prepare for a plot
#x=[]
#y=[]
#numOfConversationAnalysed=20
#for conversationNum in range(numOfConversationAnalysed):
##,data in enumerate(client.test.chatSessions.find()):
#    x.append(featureExtractor(conversationNum+5))
#    y.append(sentimentAnalyser(conversationNum+5)[0]['compound'])
#
#######################################################################
### Plot
#fig=plt.figure()
#plt.scatter(x, y, alpha=0.5)
#fig.suptitle('Conversation analysis', fontsize=20)
#plt.xlabel('Criticality', fontsize=18)
#plt.ylabel('Sentiment', fontsize=16)
#plt.show()

