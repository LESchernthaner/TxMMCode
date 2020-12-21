import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.wrappers import DtmModel
import logging
import pickle
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# Import precompiled binaries of C and C++ implementation of the DTM model
# https://github.com/magsilva/dtm/tree/master/bin
path_to_dtm_binary = "./dtm-linux64"

mainData = pd.read_pickle('./mainData.pkl')
dictionary = Dictionary.load('./mainDictionary')
corpus = mainData['corpusBOW']

'''
# Empty abstracts were filtered but empty BOW representations were NOT
unfilteredCorpus = mainData['corpusBOW']
nonEmpty = unfilteredCorpus.str.len() > 0
nonEmpty.value_counts()  # True = 20666 / False = 2062
corpus = mainData[mainData['corpusBOW'].str.len() > 0]
'''

artcCounts = mainData[['Year', 'Abstract']].groupby(['Year']).count()
timeSlice = artcCounts['Abstract'].tolist()
noTopics = [41, 44, 47, 50, 53, 56, 59, 62, 65]


# Get Umass coherence scores for model with n topics and save model
def getCoherenceScores(nTopics):
    model = DtmModel(path_to_dtm_binary, corpus=corpus, num_topics=nTopics, id2word=dictionary, time_slices=timeSlice)
    model.save(f'./Models/model{nTopics}Topics')
    wordRepresentationTopics = [model.dtm_coherence(time=time) for time in range(0, len(timeSlice))]
    coherenceModels = [CoherenceModel(topics=wordRepresentationTopics[time],
                                      corpus=corpus,
                                      dictionary=dictionary,
                                      coherence='u_mass') for time in range(0, len(timeSlice))]
    coherenceScores = [coherenceModels[time].get_coherence() for time in range(0, len(timeSlice))]
    return coherenceScores


modelScores = {}
for amount in noTopics:
    print(f'Modeling {amount} topics')
    print(f'Current time: {datetime.now().hour}:{datetime.now().minute}')
    scoresAmountTopics = getCoherenceScores(amount)
    modelScores[f'scores{amount}Topics'] = scoresAmountTopics

with open('./Models/modelScores', 'wb') as f:
    pickle.dump(modelScores, f)
