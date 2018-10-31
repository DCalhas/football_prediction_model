#!/usr/bin/env python
import csv
import numpy as np


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from sklearn.preprocessing import MaxAbsScaler

from sklearn.decomposition import PCA, KernelPCA

from sklearn.cluster import KMeans


from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

def getDataFromCSV(file_name, dir_name):
    games = []

    with open(dir_name +  "/" + file_name, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            row[2] = int(row[2])
            row[3] = int(row[3])
            games += [row]

    return games


# ## Compute how many goals difference each team has

# In[2]:

def getGoalsDifferences(teams, games):
    goal_difference = dict(zip(teams, [0]*len(teams)))
    goals_scored = dict(zip(teams, [0]*len(teams)))
    goals_suffered = dict(zip(teams, [0]*len(teams)))
    home_goals_scored = dict(zip(teams, [0]*len(teams)))
    home_goals_suffered = dict(zip(teams, [0]*len(teams)))
    away_goals_scored = dict(zip(teams, [0]*len(teams)))
    away_goals_suffered = dict(zip(teams, [0]*len(teams)))

    away_goal_difference = dict(zip(teams, [0]*len(teams)))
    home_goal_difference = dict(zip(teams, [0]*len(teams)))

    for g in games:
        goal_difference[g[0]] += g[2] - g[3]
        goal_difference[g[1]] += g[3] - g[2]
        
        goals_scored[g[0]] += g[2]
        goals_suffered[g[0]] += g[3]
        goals_scored[g[1]] += g[3]
        goals_suffered[g[1]] += g[2]
        
        home_goals_scored[g[0]] += g[2]
        home_goals_suffered[g[0]] += g[3]
        away_goals_scored[g[1]] += g[3]
        away_goals_suffered[g[1]] += g[2]
        
        away_goal_difference[g[1]] += g[3] - g[2]
        home_goal_difference[g[0]] += g[2] - g[3]
    features = []

    for i in range(len(goal_difference.values())):
        features += [[list(goal_difference.values())[i],
                     list(goals_scored.values())[i],
                    list(goals_suffered.values())[i],
                    list(home_goal_difference.values())[i],
                    list(home_goals_scored.values())[i],
                    list(home_goals_suffered.values())[i],
                    list(away_goal_difference.values())[i],
                    list(away_goals_scored.values())[i],
                    list(away_goals_suffered.values())[i]]]


    features = np.array(features)
    return features
        

# ## Clustering

# Applying PCA dimensionality reduction for 99% explainability of the data

# In[6]:


def normalize(features):


    max_abs_scaler = MaxAbsScaler()
    features = max_abs_scaler.fit_transform(features)

    return features

def pca_features(features, teams, games):
    

    pca = PCA(n_components=0.99) #number of components to explain 99% of the data
    features_pca = pca.fit_transform(features)
    print(features_pca)

    data_pca_features = []

    for g in range(len(games)):
        game = []
        
        game += features_pca[teams.index(games[g][0])].tolist() + features_pca[teams.index(games[g][1])].tolist()
        
        
        if(games[g][2] - games[g][3] > 0):
            game += [1]
        elif(games[g][2] - games[g][3] < 0):
            game += [-1]
        else: 
            game += [0]

        #game += [games[g][2] - games[g][3]]
        
        data_pca_features += [game]

    data_pca_features = np.array(data_pca_features)

    return data_pca_features


# Apllying Kmeans algorithm for k in range [2,10]

# In[8]:


def kmeans_features(features, teams, games):
    

    kmeans = []
    kmeans_predictions = []
    for k in range(0, 9):
        kmeans += [KMeans(n_clusters=k+2, random_state=170)]
        
        kmeans[k].fit(features)
        
        kmeans_predictions += [kmeans[k].predict(features)]
        print("Score of k =", k+2, " is: ", kmeans[k].score(features))

    data_kmeans_features = []

    for g in range(len(games)):
        game = []
        
        game += [kmeans_predictions[4][teams.index(games[g][0])]]
        game += [kmeans_predictions[4][teams.index(games[g][1])]]
        
        
        if(games[g][2] - games[g][3] > 0):
            game += [1]
        elif(games[g][2] - games[g][3] < 0):
            game += [-1]
        else: 
            game += [0]
        
        #game += [games[g][2] - games[g][3]]
        
        data_kmeans_features += [game]

    data_kmeans_features = np.array(data_kmeans_features)
    return data_kmeans_features


def all_Features(features, teams, games):

    data_all_features = []

    for g in range(len(games)):
        game = []
        
        game += features[teams.index(games[g][0])].tolist() + features[teams.index(games[g][1])].tolist()
        
        
        if(games[g][2] - games[g][3] > 0):
            game += [1]
        elif(games[g][2] - games[g][3] < 0):
            game += [-1]
        else: 
            game += [0]

        #game += [games[g][2] - games[g][3]]
        
        data_all_features += [game]

    data_all_features = np.array(data_all_features)
    return data_all_features


# # Support Vector Machine classifier to predict game result

# In[20]:

def performSVC(features):

    

    # spit the data into train and test set
    X = features[:,range(len(features[0])-1)]
    y = features[:,len(features[0])-1]

    for i in range(len(y)):
        if(y[i] != 1):
            y[i] = 0


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #Test linear kernel

    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)




    y_train_pred = clf.predict(X_train)

    print("Train set prediction metrics\n\n\n")
    print("Mean absolute error: ", mean_absolute_error(y_train, y_train_pred))
    print("Accuracy score: ", accuracy_score(y_train, y_train_pred))
    print("Precision score: ", precision_score(y_train, y_train_pred, average='macro'))
    print("Recall score: ", recall_score(y_train, y_train_pred, average='macro'))

    y_test_pred = clf.predict(X_test)

    print("\n\n\nTest set prediction metrics\n\n\n")
    print("Mean absolute error: ", mean_absolute_error(y_test, y_test_pred))
    print("Accuracy score: ", accuracy_score(y_test, y_test_pred))
    print("Precision score: ", precision_score(y_test, y_test_pred, average='macro'))
    print("Recall score: ", recall_score(y_test, y_test_pred, average='macro'))


    print("Confusion matrix for test set")
    print(confusion_matrix(y_test, y_test_pred))


    print("Confusion matrix for train set")
    print(confusion_matrix(y_train, y_train_pred))

    return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred


