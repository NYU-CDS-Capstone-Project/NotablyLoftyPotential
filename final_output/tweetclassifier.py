import os
import sys
import string
import pickle
import tarfile
from gensim.models import ldamodel, Phrases, phrases
from gensim import corpora
from nltk.corpus import stopwords

class classifytweet:
    def __intit__(self, tweet_body='tweet_body'):
        """
        tweet_body: the individual tweet
        model_files: the location for the model files. leave the model files folder as is
                     unless you have changed the location of all the model files.
        """
        self.tweet = tweet_body
        self.model = None
        self.corpus = None
        self.dictionary = None

    def initialize(self):
        """
        Initialize the model.
        """
        self.model, self.corpus, self.dictionary, self.phraser = load_model()
        self.lemmatized, self.stemmed, self.phrased = prepare_tweet()

    def load_model(self, model_files='model_files/'):
        """
        Loads the model, corpus, and dictionary.
        """
        # extract tarfile
        for f in os.listdir(model_files):
            if f.endswith('.gz'):
                tar = tarfile.open(model_files + f, "r:gz")
                tar.extractall(path=model_files)
                tar.close()

        # load model, corpus, and dictionary objects
        fnames = [fn for fn in os.listdir(model_files) if '.gensim' in fn]
        model = ldamodel.LdaModel.load(model_files + fnames[0])
        corpus = corpora.MmCorpus(model_files + 'unigrams_corpus.mm')
        dictionary = corpora.Dictionary.load(model_files + 'unigrams_dictionary.pkl')
        model.id2word = dictionary
        phraser = phrases.load(model_files+'document_phraser.pkl')

        #cleanup the unzipped files
        for f in fnames:
            os.remove(f)
        return model, corpus, dictionary, phraser

    def prepare_tweet(self):
        """
        Turn that unstructured text into sweet, sweet, "cleaned" up tokens!
        """

        lemmatized = # TO DO

        stemmed = # TO DO

        keep = set(['!','?'])
        stop = set(stopwords.words('english'))
        remove = set([x for x in list(string.punctuation) if x not in keep])
        stop.update(remove)
        stop.update(['',' ','  '])
        stemmed = [d for d in self.tweet.split(" ") if d not in stop] # NEEDS TO UPDATED FOR PORTER STEMMER-ING THE TWEET
        phrased = self.phraser([stemmed])

    return lemmatized, stemmed, phrased

    def valence_arousal(self):
        """
        Creates the valence and arousal score for the tweet.
        """

        return valence_score, arousal_score

    def sentiment_score(self):
        """
        Weights the posititive/negative sentiment of the tweet.
        """

        return sentiment_score

    def get_topics(self):
        """
        Extract the topics from the tweet using the LDA model.
        """

        return topic_list_of_tuples

    def emoji_cout(self):
        """
        Count the Mad! faces.
        """

        return emoji_count

    def outrage_count(self):
        """
        Get the number of outrage words in the tweet.
        """

        return outrageousness


    def outrage_score(self):
        """
        Uses the results of each of the index measures to create one score.
        """
        self.va = valence_arousal()
        self.sentiment = sentiment_score()
        self.topics = get_topics()
        self.emjois = emoji_count()
        self.outrageousness = outrage_count()

        outrageous_meter = # DO SOME SHIT!

        return outrageous_meter




if __name__ == '__main__':

    tweet_body = sys.argv[1] # the body of the tweet
    classifier = classifytweet(tweet_body)
    classifier.initialize()
