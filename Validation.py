import pandas as pd
import gensim
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

sns.set()

nltk.download('stopwords')
nltk.download('wordnet')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 85)

validationData = pd.read_pickle('./validationData_unprocessed.pkl')
years = validationData['Year'].unique().tolist()
print(f'Years: {years}')
artCounts = validationData[['Year', 'Abstract']].groupby(['Year']).count()
noArtYear = artCounts['Abstract'].tolist()
print(f'Time slice sizes: {noArtYear}')


# https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
# Split into words, remove short words, accents and punctuation marks
# Remove stop words, lemmatize and stem
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text, deacc=True, min_len=3):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result


PrePAbstracts = validationData['Abstract'].map(preprocess)
validationData['PrePAbstracts'] = PrePAbstracts

# https://medium.com/@kurtsenol21/topic-modeling-lda-mallet-implementation-in-python-part-1-c493a5297ad2
# Create bag-of-words representation of the dictionary
dictionary = gensim.corpora.Dictionary(validationData['PrePAbstracts'])
print(f'Length initial dictionary: {len(dictionary)}')

# Remove words with frequency higher than 4000
dict_words = [dictionary.doc2bow(text) for text in PrePAbstracts]
dict_corpus = {}

for i in range(len(dict_words)):
    for idx, freq in dict_words[i]:
        if dictionary[idx] in dict_corpus:
            dict_corpus[dictionary[idx]] += freq
        else:
            dict_corpus[dictionary[idx]] = freq

dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])

plt.figure(figsize=(8, 6))
sns.distplot(dict_df['freq'], bins=100)
plt.show()

print('Top 10 words in dictionary: ')
print(dict_df.sort_values('freq', ascending=False).head(10))

maxFreq = 4000
extension = dict_df[dict_df.freq > maxFreq].index.tolist()
ids = [dictionary.token2id[extension[i]] for i in range(len(extension))]
dictionary.filter_tokens(bad_ids=ids)
print(f'Dictionary length after filtering out >{maxFreq}: {len(dictionary)}')

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
print(f'Final length of dictionary: {len(dictionary)}')

# Printing some common terms to check if dictionary is constructed correctly
print('Term autism present in filtered dictionary?')
print('autism' in dictionary.token2id)

print('Term asd present in filtered dictionary?')
print('asd' in dictionary.token2id)

print('Term children present in filtered dictionary?')
print('children' in dictionary.token2id)

print('Term disord present in filtered dictionary?')
print('disord' in dictionary.token2id)

# Create bag-of-words representation
bowRepresentation = [dictionary.doc2bow(doc) for doc in validationData['PrePAbstracts']]
validationData['corpusBOW'] = bowRepresentation

# Remove abstracts with empty BOW representations
validationData = validationData[validationData['corpusBOW'].str.len() > 0]
# validationData = validationData.reset_index(drop=True)
print(f'Final length validation corpus: {len(validationData)}')

# Save final dataframe and dictionary
validationData.to_pickle('./validationData.pkl')
dictionary.save('./validationDictionary')

# Get terms in decreasing term- and document frequency
topn = 50
sortedTFKeys = sorted(dictionary.cfs, reverse=True, key=lambda i: int(dictionary.cfs[i]))
sortedTF = [(dictionary[x], dictionary.cfs[x]) for x in sortedTFKeys]
print(f'Top {topn} terms with highest term frequency:')
pprint.pprint(sortedTF[0:topn])

sortedDFKeys = sorted(dictionary.dfs, reverse=True, key=lambda i: int(dictionary.dfs[i]))
sortedDF = [(dictionary[x], dictionary.dfs[x]) for x in sortedDFKeys]
print(f'Top {topn} terms with highest document frequency:')
pprint.pprint(sortedDF[0:topn])
