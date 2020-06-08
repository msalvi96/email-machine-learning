import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from time import sleep

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from helper_functions import *

# emails = pd.read_csv('split_emails.csv')
# email_df = pd.DataFrame(parse_emails(emails.message))

# email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)

#tokenize email bodies and convert to document term matrix
# stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
# vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)

# X = vect.fit_transform(email_df.body)
# features = vect.get_feature_names()
# X_dense = X.todense()
# coords = PCA(n_components=2).fit_transform(X_dense)
# plt.scatter(coords[:, 0], coords[:, 1], c='m')
# plt.show()

# n_clusters = 3
# clf = KMeans(n_clusters=n_clusters, 
#             max_iter=100, 
#             init='k-means++', 
#             n_init=1)
# labels = clf.fit_predict(X)

# X_dense = X.todense()
# pca = PCA(n_components=2).fit(X_dense)
# coords = pca.transform(X_dense)

# centroids = clf.cluster_centers_
# centroid_coords = pca.transform(centroids)
# label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
#                 "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
# colors = [label_colors[i] for i in labels]
# plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
# plt.show()
# plt.scatter(coords[:, 0], coords[:, 1], c=colors)
# plt.show()

# print(top_feats_in_doc(X, features, 1, 10))

# print(email_df.head())
# plot_tfidf_classfeats_h(top_feats_per_cluster(X, labels, features, 0.1, 25))



def get_int_input():
    """ Function to get integer input """

    var_input = input()
    try:
        var_input = int(var_input)
        return var_input
    except ValueError:
        print('ERROR: Enter a Valid Number... \n')
        get_int_input()

def simplePCA(X):
    X_dense = X.todense()
    coords = PCA(n_components=2).fit_transform(X_dense)
    plt.scatter(coords[:, 0], coords[:, 1], c='m')
    plt.show()

def topFeatures(X, features):
    print("Enter the number of important words you want to retrieve:")
    number = get_int_input()
    print(top_mean_feats(X, features, top_n=number))

def kmeansCluster(X):
    n_clusters = 3
    clf = KMeans(n_clusters=n_clusters, 
                max_iter=100, 
                init='k-means++', 
                n_init=1)
    labels = clf.fit_predict(X)
    X_dense = X.todense()
    pca = PCA(n_components=2).fit(X_dense)
    coords = pca.transform(X_dense)

    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
    colors = [label_colors[i] for i in labels]
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
    plt.show()

def display_ui(X, features):

    print_list = [
        ["Get important words from email dataset!", 111],
        ["Simple Principle Component Analysis of email dataset", 222],
        ["K-Means Clustering - Unsupervised ML Model to classify emails", 333]
    ]

    display_table = PrettyTable()
    display_table.field_names = ["Description", "Command"]
    for row in print_list:
        display_table.add_row(row)
    
    print("Welcome! This is a simple machine learning application that produces insights on email data!")
    print(display_table)
    print("Enter command number to test the feature:")
    choice = get_int_input()
    if choice == 111:
        topFeatures(X, features)

    elif choice == 222:
        simplePCA(X)

    elif choice == 333:
        kmeansCluster(X)

    else:
        raise SystemExit

if __name__ == "__main__":
    emails = pd.read_csv('split_emails.csv')
    email_df = pd.DataFrame(parse_emails(emails.message))

    email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)
    stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
    vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)

    X = vect.fit_transform(email_df.body)
    features = vect.get_feature_names()

    try:
        display_ui(X, features)

    except SystemExit:
        print("Thank you! - Mrunal Salvi")
        sleep(5)
