import os
import sys
import string
import pickle
import tarfile
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import ldamodel, Phrases, phrases
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer


class classifytweet:
    def __init__(self, model_files='model_files/'):
        """
        tweet_body: the individual tweet
        model_files: the location for the model files. leave the model files folder as is
                     unless you have changed the location of all the model files.
        """
        self.model_files = model_files
        #self.model, self.corpus, self.dictionary, self.phraser, self.valence_mean, self.arousal_mean, self.valence_sd, self.arousal_sd = self.load_model()
        self.load_model()
        self.n = len(self.dictionary.items())
        self.word_map = {v:k for k,v in self.dictionary.items()}
        print("Model items loaded and classifier initialized!")

    def load_model(self):
        """
        Loads the model, corpus, and dictionary.
        """
        # extract tarfiles
        for f in os.listdir(self.model_files):
            if f.endswith('.gz'):
                tar = tarfile.open(self.model_files + f, "r:gz")
                tar.extractall(path=self.model_files)
                tar.close()

        # load model, corpus, and dictionary objects
        fnames = [fn for fn in os.listdir(self.model_files) if '.gensim' in fn]
        self.model = ldamodel.LdaModel.load(self.model_files + fnames[0])
        self.corpus = corpora.MmCorpus(self.model_files + 'unigrams_corpus.mm')
        self.dictionary = corpora.Dictionary.load(self.model_files + 'unigrams_dictionary.pkl')
        self.model.id2word = self.dictionary
        self.phraser = phrases.Phrases.load(self.model_files + 'document_phraser.pkl')
        for f in ['unigrams_dictionary.pkl', 'unigrams_corpus.mm', 'unigrams_corpus.mm.index', 'NB_vectorizer.pkl', 'NB_sentiment_model.pkl']:
            fnames.append(f)

        # load the valence and arousal arrays
        self.valence_mean = pickle.load(open(self.model_files + 'valence_mean.pkl', 'rb'))
        self.arousal_mean = pickle.load(open(self.model_files + 'arousal_mean.pkl', 'rb'))
        self.valence_sd = pickle.load(open(self.model_files + 'valence_sd.pkl', 'rb'))
        self.arousal_sd = pickle.load(open(self.model_files + 'arousal_sd.pkl', 'rb'))

        # load the MinMaxScaler for the transforming the scores
        self.base_outrage_scaler = None
        self.expanded_outrage_scaler = None
        self.valence_scaler = None
        self.arousal_scaler = None
        self.emoji_scaler = None
        self.topic_valence_scaler = pickle.load(open(self.model_files + 'topic_valence_scaled.pkl', 'rb'))
        self.topic_arousal_scaler = pickle.load(open(self.model_files + 'topic_arousal_scaled.pkl', 'rb'))

        # load the Naive Bayes sentiment model
        try:
            self.nb_model = pickle.load(open(self.model_files + 'NB_sentiment_model.pkl', 'rb'))
        except:
            self.nb_model = pickle.load(open(self.model_files + 'NB_sentiment_model.pkl', 'rb'), encoding='latin1')
        try:
            self.nb_vectorizer = pickle.load(open(self.model_files + 'NB_vectorizer.pkl', 'rb'))
        except:
            self.nb_vectorizer = pickle.load(open(self.model_files + 'NB_vectorizer.pkl', 'rb'), encoding='latin1')

        # load the outrage dictionaries
        self.outrage_list = pd.read_csv(self.model_files + 'outrage_dictionary_stemmed.csv', header=None)
        self.exp_outrage_list = pd.read_csv(self.model_files + 'expanded_outrage_dictionary_stemmed.csv', header=None)

        # cleanup the unzipped files
        for f in fnames:
            os.remove(self.model_files + f)

    def prepare_tweet(self, tweet):
        """
        Turn that unstructured text into sweet, sweet, "cleaned" up tokens!
        """
        self.tweet = tweet
        stemmer = SnowballStemmer("english")
        tokenizer = TweetTokenizer()
        self.tweet_tokenized = tokenizer.tokenize(self.tweet)
        try:
            self.tweet_tokenized = [unicode(y.encode("utf-8"), errors='ignore') for y in self.tweet_tokenized]
            self.stemmed = [stemmer.stem(y) for y in self.tweet_tokenized]
        except:
            #tweet_tokenized = [y.encode("utf-8") for y in tweet_tokenized]
            self.stemmed = [stemmer.stem(y) for y in self.tweet_tokenized]

        #self.text_bigrams = [' '.join(self.stemmed[i:]) for i in range(2)]
        #self.text_bigrams=list(bigrams(self.stemmed))
        #self.text_bigrams=["%s %s" % x for x in self.text_bigrams]
        #self.text_bigrams.extend(self.stemmed)

        keep = set(['!','?'])
        stop = set(stopwords.words('english'))
        remove = set([x for x in list(string.punctuation) if x not in keep])
        stop.update(remove)
        stop.update(['',' ','  '])
        stemmed = [d for d in self.stemmed if d not in stop]
        self.phrased = list(self.phraser[[stemmed]])[0]

        #print('Phrased representation: "' + ' '.join(self.phrased) + '"')
        #return None

    def get_valence_score(self):
        """
        Creates the valence and arousal score for the tweet.
        """
        tweet_arr = np.zeros(self.n)
        for word in set(self.phrased) & set(self.word_map.keys()):
            tweet_arr[self.word_map[word]] = 1.
        mean = tweet_arr * self.valence_mean
        sd = tweet_arr * self.valence_sd
        total_sd = np.sum(sd) * tweet_arr
        with np.errstate(divide='ignore'):
            sd_ratio = total_sd / sd
            sd_ratio[sd == 0] = 0
        sd_weight = sd_ratio / np.sum(sd_ratio)

        if np.sum(mean*sd_weight) == np.nan:
            self.valence_score = 0
        else:
            self.valence_score = np.sum(mean*sd_weight)

        return self.valence_score

    def get_arousal_score(self):
        """
        Creates the valence and arousal score for the tweet.
        """
        tweet_arr = np.zeros(self.n)
        for word in set(self.phrased) & set(self.word_map.keys()):
            tweet_arr[self.word_map[word]] = 1.
        mean = tweet_arr * self.arousal_mean
        sd = tweet_arr * self.arousal_sd
        total_sd = np.sum(sd) * tweet_arr
        with np.errstate(divide='ignore'):
            sd_ratio = total_sd / sd
            sd_ratio[sd == 0] = 0
        sd_weight = sd_ratio / np.sum(sd_ratio)

        if np.sum(mean*sd_weight) == np.nan:
            self.arousal_score = 0
        else:
            self.arousal_score = np.sum(mean*sd_weight)

        return self.arousal_score

    def get_sentiment_score(self):
        """
        Weights the posititive/negative sentiment of the tweet.
        """
        vectorized = self.nb_vectorizer.transform(self.stemmed)
        self.sentiment_score = np.average(1 - self.nb_model.predict_proba(vectorized)[:,1])

        return self.sentiment_score

    def get_topics(self):
        """
        Extract the topics from the tweet using the LDA model.
        """
        return self.model.get_document_topics(self.model.id2word.doc2bow(self.phrased), per_word_topics=False)

    def get_emoji_count(self):
        """
        Count the Mad! faces.
        """
        positives = ['\<f0\>\<U\+009F\>\<U\+0099\>\<U\+0082\>']
        outrage = ['\<f0\>\<U\+009F\>\<U\+0098\>\<U\+00A4\>', '\<f0\>\<U\+009F\>\<U\+0098\>\<U\+00A0\>', \
                '\<f0\>\<U\+009F\>\<U\+0098\>\<U\+00A1\>']
        positive_score= self.tweet.str.contains('|'.join(positives)).astype(int)
        outrage_score= self.tweet.str.contains('|'.join(outrage)).astype(int)
        self.emoji_count = outrage_score-positive_score
        return self.emoji_count

    def get_base_outrage_count(self):
        """
        Get the number of outrage words in the tweet.
        """
        self.base_outrage_count = 0
        for i in self.stemmed:
            self.base_outrage_count += len(set(i) & set(self.outrage_list))
        return self.base_outrage_count

    def get_expanded_outrage_count(self):
        """
        Get the number of outrage words in the tweet.
        """
        self.expanded_outrage_count = 0
        for i in self.stemmed:
            self.expanded_outrage_count += len(set(i) & set(self.exp_outrage_list))
        return self.expanded_outrage_count

    def get_outrage_score(self):
        """
        Uses the results of each of the index measures to create one score.
        .12 outrage dict
        .10 expanded outrage dict
        .14 arousal
        .14 valence
        .11 sentiment
        .10 emoji
        .15 topic valence
        .14 topic arousal
        """
        self.topics = self.get_topics()
        topic_valence_score = 0
        topic_arousal_score = 0
        for tup in self.topics:
            topic_valence_score += self.topic_valence_scaler[tup[0]] * tup[1]
            topic_arousal_score += self.topic_valence_scaler[tup[0]] * tup[1]

        scores = np.array([
            self.get_base_outrage_count(),
            self.get_expanded_outrage_count(),
            self.get_arousal_score(),
            self.get_valence_score(),
            self.get_sentiment_score(),
            self.get_emoji_count(),
            topic_valence_score,
            topic_arousal_score
            ])
        weights = np.array([0.12, 0.10, 0.14, 0.14, 0.11, 0.10, 0.15, 0.14])

        self.outrage_meter = np.sum(scores*weights)
        return self.outrage_meter
