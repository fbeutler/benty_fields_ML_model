'''
Copyright 2017, Florian Beutler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

###########################################################################

This code is used to make paper recommendations for users on www.benty-fields.com.
It has two use cases:

(1) It is used to order the daily new publications for each user according
    to the user's interest
(2) It is used to provide paper recommendations for each user

If you have suggestions for improvements please contact benty-fields@feedback.com.

###########################################################################
'''
import os
import re
import math
import numpy as np
import nltk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

# path to the folder with feature data
base_path = 'path/to/folder'


def test_paper_recommendation_engine():
    '''
    This function tests the paper recommendation engine
    '''
    filename1 = base_path + "/AUC_vs_data_cmp.dat"
    with open(filename1, "w") as f:
        f.write("# data_size AUC sig_{AUC}\n")
    filename2 = base_path + "/precision_vs_data_cmp.dat"
    with open(filename2, "w") as f:
        f.write("# data_size precision sig_{precision}\n")

    for i in range(0, 1403):
        print("i = ", i)
        results = _get_model_metrics_through_cross_validation(i)

        with open(filename1, "a+") as f:
            f.write("%d %f %f\n" % (i, np.mean(results['aucs']), np.std(results['aucs'])))
        f.close()
        with open(filename2, "a+") as f:
            f.write("%d %f %f\n" % (i, np.mean(results['pr']), np.std(results['pr'])))
        f.close()
    return


def _pre_precess_text(list_of_texts):
    ''' This function cleans the abstract and title before feature selection '''
    output_list_of_texts = []
    for text in list_of_texts:
        output_list_of_texts.append(' '.join(tokenize(text)))
    return output_list_of_texts


def tokenize(text):
    ''' Remove numbers, single letters, punctuations, tokenize and stem '''
    # remove numbers
    text_without_numbers = re.sub(r'\d+', '', text)
    # remove single letters
    text_without_single_letters = re.sub(r'\b\w\b', ' ', text_without_numbers)
    # remove punctuations 
    text_without_punctuation = ' '.join(re.findall('\w+', text_without_single_letters))
    tokens = nltk.word_tokenize(text_without_punctuation)
    #!!!!! we removed stemming, since this is very slow !!!!!!!!
    #stems = stem_tokens(tokens, stemmer)
    #return stems
    return tokens


def _get_model_metrics_through_cross_validation(ref_id):
    ''' 
    This function calculates different metrics for the paper recommendation model
    including a cross-validation error estimate
    '''
    paper_id_file = base_path + "/arxiv_ids_%d.gz" % ref_id
    abstract_features_file = base_path + "/abstract_features_%d.gz" % ref_id
    title_features_file = base_path + "/title_features_%d.gz" % ref_id
    author_features_file = base_path + "/author_features_%d.gz" % ref_id
    y_file = base_path + "/y_%d.gz" % ref_id

    # Instead of loading the features here you could use the arxiv API and
    # get a whole lot of other metadata (https://arxiv.org/help/api)
    list_of_paper_ids = np.loadtxt(paper_id_file, dtype=str, delimiter=' ')
    list_of_abstracts = np.genfromtxt(abstract_features_file, dtype=str, delimiter='\t')
    list_of_titles = np.loadtxt(title_features_file, dtype=str, delimiter='\t')
    list_of_authors = np.loadtxt(author_features_file, dtype=str, delimiter='\t')
    y = np.loadtxt(y_file, dtype=int, delimiter=' ')

    X1, X2, X3 = _prepare_data(list_of_abstracts, list_of_titles, list_of_authors)
    X = np.c_[X1, X2, X3]

    metrics = {
        'num_cases': len(X),
        'curves': [],
        'aucs': [],
        'pr': []
    }
    cross_validation_steps = 10
    kf = KFold(n_splits=cross_validation_steps, shuffle=True)
    forest = RandomForestClassifier(n_estimators=100)

    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        forest.fit(X_train, y_train)
        probabilities = forest.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, probabilities, pos_label=1)

        thresholds = np.append(thresholds, np.array([1]))

        false_positive_rate, true_positive_rate, thresholds2 = roc_curve(y_test, probabilities, pos_label=1)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print("roc_auc = ", roc_auc)
        if not math.isnan(roc_auc):
            av_pr = average_precision_score(y_test, probabilities)

            case_rate = []
            for threshold in thresholds:
                case_rate.append(np.mean(probabilities >= threshold))

            curves = {
                'thresholds': thresholds,
                'precision': precision,
                'recall': recall,
                'case_rate': case_rate,
                'fpr': false_positive_rate,
                'tpr': true_positive_rate
            }
            metrics['curves'].append(curves)
            metrics['aucs'].append(roc_auc)
            metrics['pr'].append(av_pr)
    _plot_cross_validation_result(ref_id, metrics)
    return metrics


def _prepare_data(list_of_abstracts, list_of_titles, list_of_authors):
    '''
    Here we create term frequency-inverse
    document frequency (tf-idf) features.
    '''
    # We pre-process the abstracts and titles, including stemming
    list_of_abstracts = _pre_precess_text(list_of_abstracts)
    list_of_titles = _pre_precess_text(list_of_titles)

    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 stop_words="english",
                                 max_features=3000)
    # fit_transform() serves two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    abstract_features = vectorizer.fit_transform(list_of_abstracts)
    title_features = vectorizer.fit_transform(list_of_titles)
    author_features = vectorizer.fit_transform(list_of_authors)

    # Numpy arrays are easy to work with, so convert the results to arrays
    abstract_features = abstract_features.toarray()
    title_features = title_features.toarray()
    author_features = author_features.toarray()
    return abstract_features, title_features, author_features


def _plot_cross_validation_result(ref_id, results):
    ''' ROC Curve plot '''
    # average values for the legend
    auc_av = "{0:.2f}".format(np.mean(results['aucs']))
    auc_sd = "{0:.2f}".format(np.std(results['aucs']))

    plt.clf()
    plt.figure(2)
    # plot each individual ROC curve
    for chart in results['curves']:
        plt.plot(chart['fpr'], chart['tpr'], color='b', alpha=0.5)
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Cross-validated ROC for user %d' % (ref_id))
    plt.text(0.6, 0.1, r'AUC = {av} \pm {sd}'.format(av=auc_av, sd=auc_sd))
    plt.savefig(base_path + "/user_like_roc_curve_%d.png" % ref_id)

    # Precision-recall plot
    pr_av = "{0:.2f}".format(np.mean(results['pr']))
    pr_sd = "{0:.2f}".format(np.std(results['pr']))

    plt.clf()
    plt.figure(4)
    for chart in results['curves']:
        plt.plot(chart['recall'], chart['precision'], color='b', alpha=0.5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Cross-validated precision/recall for user %d' % (ref_id))
    plt.text(0.6, 0.9, r'AUC = {av} \pm {sd}'.format(av=pr_av, sd=pr_sd))
    plt.savefig(base_path + "/user_like_precision_recall_%d.png" % ref_id)
    return
