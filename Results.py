import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.wrappers import DtmModel
import pickle
from wordcloud import WordCloud
import pprint

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 250)


# Load model with specified amount of topics
def loadDTM(amounttopics):
    modelAmountTopics = DtmModel.load(f'./Models/model{amounttopics}Topics')
    return modelAmountTopics


# Load scores of models with different numbers of topics and the corpus used for training (BOW representations)
def loadScoresCorpus():
    mainData = pd.read_pickle('./mainData.pkl')
    mainData = mainData[mainData['corpusBOW'].str.len() > 0]
    mainData = mainData.reset_index(drop=True)
    years = mainData['Year'].unique().tolist()
    corpus = mainData['corpusBOW']

    with open('./Models/modelScores', 'rb') as f:
        modelScores = pickle.load(f)

    modelscoresdf = pd.DataFrame.from_dict(modelScores)
    modelscoresdf['Years'] = years
    return modelscoresdf, corpus


# Plot coherence scores over time for all models
def plotCoherenceScores(modelscoresdf, colors):
    noTopics = [41, 44, 47, 50, 53, 56, 59, 62, 65]

    plt.clf()
    plt.figure(figsize=(6, 5))
    indexColors = 0
    for amount in noTopics:
        plt.plot('Years', f'scores{amount}Topics', data=modelscoresdf, marker='', color=colors[indexColors],
                 label=amount)
        indexColors += 1

    plt.legend()
    locs, labels = plt.yticks()
    plt.xticks(range(2009, 2019))
    plt.xlabel('Year')
    plt.ylabel('UMass coherence score')
    for tick in locs:
        plt.axhline(y=tick, alpha=0.5, linewidth=0.5, linestyle='dashed')

    plt.tight_layout(pad=0)
    plt.savefig('./Models/modelScores.png', bbox_inches='tight')
    plt.show()
    return


# Test run to obtain top n words in given topic at given time
def topWordsTopic(model, timeid, topicid, topn):
    topicsTest = model.show_topic(topicid=topicid - 1, time=timeid - 1, topn=topn)
    pprint.pprint(
        f'Top {topn} words and their probability for topic id {topicid} at time {timeid}: {topicsTest}')
    return


# https://amueller.github.io/word_cloud/auto_examples/frequency.html
# Get frequency dictionary of words in topic at time
def getFreqDict(model, topn, timeid, topicid):
    wordsList = model.show_topic(topicid=topicid, time=timeid, topn=topn)
    wordsList = dict([(entry[1], entry[0]) for entry in wordsList])
    return wordsList


# Create a word cloud for the given list of words and their frequencies
# then save or plot the result
def makeWordNet(wordslist, timeid, topicid, save):
    wc = WordCloud(width=500, height=350).generate_from_frequencies(wordslist)
    plt.clf()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    if save:
        plt.savefig(f'./WordClouds/wordcloudTime{timeid}Topic{topicid}.png', bbox_inches='tight')
    else:
        plt.show()
    return


# Make word clouds of top 30 words for 41 topics model for
# 1. 7 interesting topics (or the first 7 topics if none given)
# 2. first topic at first 7 time intervals
def wordCloudAnalysis(model, interestingTopics, topn, save):
    topicIds = [topicId - 1 for topicId in interestingTopics]

    for topicId in topicIds:
        wordsList = getFreqDict(model=model, topn=topn, timeid=0, topicid=topicId)
        makeWordNet(wordslist=wordsList, timeid=0, topicid=topicId, save=save)

    for timeId in range(0, 7):
        wordsList = getFreqDict(model=model, topn=topn, timeid=timeId, topicid=0)
        makeWordNet(wordslist=wordsList, timeid=timeId, topicid=0, save=save)

    return


# Make and show or plot a single word cloud
def singleWordCloud(model, topn, timeid, topicid, save):
    wordsList = getFreqDict(model=model, topn=topn, timeid=timeid, topicid=topicid)
    makeWordNet(wordslist=wordsList, timeid=timeid, topicid=topicid, save=save)


# Get probability of top words over time of single topic (topic 3)
def topWordProbability(model, years):
    wordsList = [getFreqDict(model=model, topn=40, timeid=year, topicid=0) for year in range(0, len(years))]
    topprobdf = pd.DataFrame(wordsList)
    return topprobdf


# Plot probability of top words over time
def plotTopWordProb(model, years, colors):
    topprobdf = topWordProbability(model, years)
    topprobdf = topprobdf.iloc[:, 0:10]
    topWords = topprobdf.columns.tolist()
    topprobdf['Years'] = years

    plt.clf()
    indexColors = 0
    for word in topWords:
        plt.plot('Years', word, data=topprobdf, marker='', color=colors[indexColors],
                 label=word)
        indexColors += 1

    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.legend()
    plt.tight_layout(pad=0)
    plt.savefig(f'./Models/topWordProbs.png', bbox_inches='tight')
    plt.show()
    return


# Get mean proportions of all topics over time
def calcFromGamma(model):
    gamma = model.gamma_
    meandtprops = np.zeros((len(model.time_slices), model.num_topics))
    prevSlice = 0
    for sliceNo, tSlice in enumerate(model.time_slices):
        gammaTime = gamma[prevSlice:prevSlice + tSlice]
        meandtprops[sliceNo] = gammaTime.mean(axis=0)
        prevSlice = prevSlice + tSlice

    return meandtprops.transpose()


# Plot topic popularities over time
def plotTopicProp(model, corpus, years):
    dtpropTime = calcFromGamma(model)
    plt.clf()
    fig, ax = plt.subplots()
    maxLabels = 10
    plotLabels = [(topicno // (maxLabels + 1)) * "_" + f'{topicno}' for topicno in range(1, len(dtpropTime) + 1)]
    ax.stackplot(years, dtpropTime, labels=plotLabels)
    plt.xticks(range(2009, 2019))
    plt.xlim(right=2019.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='lower right', borderaxespad=0.)
    plt.tight_layout(pad=0)
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.savefig(f'./Models/topicProps.png', bbox_inches='tight')
    plt.show()


# Get most popular topics on average in decreasing order
def getMeanProp(model):
    dtpop = calcFromGamma(model)
    meandtpop = dtpop.mean(axis=1)
    meandtpop = np.argsort(meandtpop)
    meandtpop = 1 + meandtpop
    meandtpop = np.flipud(meandtpop)
    print(f'Most popular topics (average, decreasing): {meandtpop}')
    return meandtpop


if __name__ == '__main__':
    [modelScoresDF, modelCorpus] = loadScoresCorpus()
    modelYears = modelScoresDF['Years'].tolist()
    plotColors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    amountTopics = 53
    model53Topics = loadDTM(amountTopics)
    amountWords = 500
    print(f'Time slices: {model53Topics.time_slices}')
    print(f'Sum time slices: {sum(model53Topics.time_slices)}')
    print(f'Corpus length: {len(modelCorpus)}')

    # plotCoherenceScores(modelscoresdf=modelScoresDF, colors=plotColors)
    # plotTopicProp(model=model53Topics, corpus=modelCorpus, years=modelYears)
    #
    # topWordsTopic(model=model53Topics, timeid=1, topicid=4, topn=20)
    # topWordsTopic(model=model53Topics, timeid=1, topicid=23, topn=20)
    # topWordsTopic(model=model53Topics, timeid=1, topicid=52, topn=20)
    #
    # wordCloudAnalysis(model=model53Topics, topn=amountWords, save=1, interestingTopics=[1,4,23,24,27,44,52])
    # singleWordCloud(model=model53Topics, topn=amountWords, timeid=0, topicid=18, save=0)
    #
    # plotTopWordProb(model=model53Topics, years=modelYears, colors=plotColors)

    topMeanProp = getMeanProp(model=model53Topics)
    topMeanProp = topMeanProp[0:5]
    for topTopidId in topMeanProp:
        topWordsTopic(model=model53Topics, timeid=1, topicid=topTopidId, topn=6)
