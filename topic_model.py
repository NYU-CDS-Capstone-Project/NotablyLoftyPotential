import os
import math
import tarfile
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import ldamodel
from gensim import corpora
from time import time

def run_lda(corpus, dictionary, alpha='auto', eta='auto', n_topics='10', num_iter='200', n_words='20', rand=10, passes=10, topic='all'):
    """
    This function is designed to take your corpus and dictionary and run the
    gensim ldamodel.LdaModel. This also outputs your model as a pickle to the
    /pickles/ directory.
    
    corups = gensim corpora
    dictionary = gensim index dictionary
    alpha = LDA hyperparameter. Keep at 'auto' to automatically learn the optimal setting.
    eta = LDA hyperparamber. Keep at 'auto' to automatically learn the optimal setting.
    n_topics = number of topics to extract
    num_iter = number of iterations
    rand = random_state
    n_words = top n words to visualize from each topic
    topic = what topic are we running with - this is just used to save the 
        viz name, so definitely change this value.
        
    Returns the model and the dataframe for visualization, and further inspection
    """
    
    directory = os.path.dirname('models/')
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    
    directory = os.path.dirname('plots/')
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    
    del directory

    # Run the model
    t0 = time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = ldamodel.LdaModel(corpus=corpus,
                              alpha=alpha,
                              eta=eta,
                              num_topics=n_topics, 
                              id2word=dictionary, 
                              iterations=num_iter, 
                              passes=passes,
                              chunksize=10000
                             )
    elapsed = time() - t0
    print (model)
    print ("Runtime: %s" %(str(datetime.timedelta(seconds=elapsed))))
    return model

def cleanup_models(model_list):
    """
    This is to clean up any open models and zip them to clear space.
    """
    for f_name, model in model_list:

        # Drop the id2word part of the model to save it later
        id2word = {k:v for k, v in model.id2word.items()}
        model.id2word = None
        model.save('models/' + f_name + '.gensim', separately=['alpha', 'eta', 'expElogbeta', 'sstats'])
        model.id2word = id2word  # restore the dictionary

        with open('models/' + f_name + '.json', 'wb') as out:
            json.dump(id2word, out)

        model.clear()
        
        # Create a tar file with all the associated model files including the JSON id2word
        file_names = [fn for fn in os.listdir('models/') if (fn.startswith(f_name)) and (fn.endswith('.json') == False)]
        os.chdir('models/')
        tar = tarfile.open(f_name + '.gz', 'w:gz')
        for f in file_names:
            tar.add(f)
        tar.close()
        
        if os.path.isfile(f_name + '.gz'):
            for f in file_names:
                os.remove(f)
        else:
            print(f_name, "did not zip. Please zip manually.")
        
        os.chdir('..')

    print("Models saved and zipped!")
    return

def load_data(corpus_file='path_to_file/file_name', dictionary_file='path_to_file/file_name', model_file='path_to_file/file_name', mmap=None):
    """
    This function is meant load previously saved models, dictionaries, and corpora.
    NOTE: all paths should be updated to reflect their actual location or it will be skipped
    nmap: the zipped format. Should only be None, '.gz', or '.bz2'.
    """
    
    return_objects = list()
    
    if corpus_file == 'path_to_file/file_name':
        print("No corpus file input. Skipping corpus loading.")
    else:
        return_objects.append(corpora.MmCorpus(corpus_file))
    
    if dictionary_file == 'path_to_file/file_name':
        print("No dictionary file input. Skipping dictionary loading.")
    else:
        return_objects.append(corpora.Dictionary.load(dictionary_file, mmap=mmap))
        
    if model_file == 'path_to_file/file_name':
        print("No model file input. Skipping model loading.")
    else:
        return_objects.append(corpora.Dictionary.load(model_file, mmap=mmap))
    
    return return_objects

def round_up_05(num):
    return math.ceil(num*20)/20

def plot_lda(model, n_topics=100, num_iter=200, n_words=20, topic='all'):
    """
    This function is designed to take your corpus and dictionary and run the
    gensim ldamodel.LdaModel. This also outputs your model as a pickle to the
    /pickles/ directory.
    
    model = output from run_lda
    n_topics = number of topics to extract
    num_iter = number of iterations
    rand = random_state
    n_words = top n words to visualize from each topic
    topic = what topic are we running with - this is just used to save the 
        viz name, so definitely change this value.
        
    Returns the model and the dataframe for visualization, and further inspection
    """
    
    # Create the data for visualization
    topicWordProbMat = model.print_topics(n_topics, num_words=n_words)
    prob_mat = dict()

    for line in topicWordProbMat:
        t, words = line
        probs = words.split(" + ")
        for pr in probs:
            word_pr = pr.strip().split("*")
            try:
                prob_mat[word_pr[1][1:-1]][t] = float(word_pr[0])
            except:
                prob_mat[word_pr[1][1:-1]] = [0.0]*n_topics
                prob_mat[word_pr[1][1:-1]][t] = float(word_pr[0])
    for k in prob_mat.keys():
        s = np.sum(prob_mat[k])
        prob_mat[k].append(s)

    df = pd.DataFrame.from_dict(prob_mat, orient='index')
    df.rename(columns={n_topics: 'sum'}, inplace=True)
    df = df.sort_values(by=['sum'], ascending=False)

    max_val = round_up_05(np.max(df.iloc[:,:n_topics].values))
    min_val = np.min([0,-np.abs(np.min(df.iloc[:,:n_topics].values))])

    # Make the visualization
    fig, ax = plt.subplots(n_topics+1, 1, figsize=(n_topics, n_words*1.3) )

    for i in range(n_topics):
        if i == 0:
            ax[i].grid(False)

            mat = [x for x in reversed(np.arange(min_val, max_val+0.05, 0.05))]
            ax[i].set_xticks(np.arange(0, len(mat), 1.0))
            ax[i].matshow(np.array(zip([0]*len(mat),mat)).T, vmin=min_val, vmax=max_val+0.05, cmap=plt.cm.hot)
            ax[i].set_xticklabels(['']+['{0:.2f}'.format(x) for x in mat], rotation=45)
            for tick in ax[i].xaxis.get_majorticklabels():
                tick.set_horizontalalignment('left')
            ax[i].set_ylim(0.5,1)
            ax[i].set_yticks([])
            ax[i].set_yticklabels([])
            plt.xticks(np.arange(0, len(mat)-1, 1.0))
            #plt.subplots_adjust(top=2)
            ax[i].set_title('Top ' + str(n_words) + ' Words per Topic by Word Probabilities', y=4, fontsize=20)

        spot = 111 + (i+1)*100
        ax[i+1].grid(False)

        mat = pd.DataFrame(df[i][df[i] != 0]).sort_values(by=i, ascending=False)
        ax[i+1].matshow(np.array(zip([0]+[0]*len(mat.values),list(mat.values))).T, vmin=min_val, vmax=max_val, cmap=plt.cm.hot)
        ax[i+1].set_xticks(np.arange(0, len(mat.values), 1.0))

        if n_topics == 25:
            ax[i+1].set_xticklabels([mat.index.values[x] + "\n"'{0:.3f}'.format(float(mat.values[x])) for x in range(len(mat.values))], rotation=45)
            ax[i+1].set_title('Topic '+str(i), y=3, fontsize=14)
        elif n_topics == 50:
            ax[i+1].set_xticklabels([mat.index.values[x] + "\n"'{0:.3f}'.format(float(mat.values[x])) for x in range(len(mat.values))], fontsize=8, rotation=35)
            ax[i+1].set_title('Topic '+str(i), y=2.7, fontsize=12)
        elif n_topics == 75:
            ax[i+1].set_xticklabels([mat.index.values[x] + "\n"'{0:.3f}'.format(float(mat.values[x])) for x in range(len(mat.values))], fontsize=6, rotation=35)
            ax[i+1].set_title('Topic '+str(i), y=2.7, fontsize=12)
        elif n_topics == 100:
            ax[i+1].set_xticklabels([mat.index.values[x] + "\n"'{0:.3f}'.format(float(mat.values[x])) for x in range(len(mat.values))], fontsize=5, rotation=35)
            ax[i+1].set_title('Topic '+str(i), y=4, fontsize=10)
        elif n_topics == 125:
            ax[i+1].set_xticklabels([mat.index.values[x] + "\n"'{0:.3f}'.format(float(mat.values[x])) for x in range(len(mat.values))], fontsize=5, rotation=35)
            ax[i+1].set_title('Topic '+str(i), y=4.5, fontsize=10)
        elif n_topics == 150:
            ax[i+1].set_xticklabels([mat.index.values[x] + "\n"'{0:.3f}'.format(float(mat.values[x])) for x in range(len(mat.values))], fontsize=5, rotation=35)
            ax[i+1].set_title('Topic '+str(i), y=5, fontsize=10)

        for tick in ax[i+1].xaxis.get_majorticklabels():
                tick.set_horizontalalignment('left')
        ax[i+1].set_ylim(0.5,1)
        ax[i+1].set_yticks([])
        ax[i+1].set_yticklabels([])

    if n_topics == 25:
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=3, wspace=None, hspace=1)
    elif n_topics == 50:
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=4, wspace=None, hspace=3)
    elif n_topics == 75:
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=4, wspace=None, hspace=3)
    elif n_topics == 100:
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=5, wspace=None, hspace=5)
    elif n_topics == 125:
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=5, wspace=None, hspace=6)
    elif n_topics == 150:
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=6, wspace=None, hspace=6)
    '''
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots
    '''
    file_path = "images/" + str(topic) + "_LDA_" + str(n_topics) + "topics_" + str(num_iter) + "_iter" + str(n_words) + "_words"+".png"
    fig.savefig(file_path, bbox_inches='tight', dpi=360)    
    #plt.show()
    
    # CLEAN UP
    del topicWordProbMat
    del prob_mat
    del df
    plt.close(fig)
    
    return None