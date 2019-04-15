import pandas as pd
import numpy as np
import sys
import platform
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import time


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
  """
  This method, takes as input the X, Y matrices of the Train and Test set.
  And fits them on all of the Classifiers specified in the dict_classifier.
  The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
  is because it is very easy to save the whole dictionary with the pickle module.

  Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
  So it is best to train them on a smaller dataset first and
  decide whether you want to comment them out or not based on the test accuracy score.
  """

  dict_models = {}
  for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
    t_start = time.clock()
    classifier.fit(X_train, Y_train)
    t_end = time.clock()

    t_diff = t_end - t_start
    train_score = classifier.score(X_train, Y_train)
    test_score = classifier.score(X_test, Y_test)

    dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                    'train_time': t_diff}
    if verbose:
      print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
  return dict_models


def display_dict_models(dict_models, sort_by='test_score'):
  cls = [key for key in dict_models.keys()]
  test_s = [dict_models[key]['test_score'] for key in cls]
  training_s = [dict_models[key]['train_score'] for key in cls]
  training_t = [dict_models[key]['train_time'] for key in cls]

  df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)),
                     columns=['classifier', 'train_score', 'test_score', 'train_time'])
  for ii in range(0, len(cls)):
    df_.loc[ii, 'classifier'] = cls[ii]
    df_.loc[ii, 'train_score'] = training_s[ii]
    df_.loc[ii, 'test_score'] = test_s[ii]
    df_.loc[ii, 'train_time'] = training_t[ii]

  print(df_.sort_values(by=sort_by, ascending=False))


if __name__=="__main__":
  # Config
  pd.set_option('display.max_columns', None)
  np.set_printoptions(threshold=sys.maxsize)

  # Data extraction and shaping
  path_train = "train_data.csv"
  path_test = "test_data.csv"

  test_dataset = pd.read_csv(path_test)
  train_dataset = pd.read_csv(path_train)

  #TRAIN DATASET
  # User_Id,Cumulative_YoutubeSess_LTE_DL_Time,Cumulative_YoutubeSess_LTE_DL_Volume,Cumulative_YoutubeSess_UMTS_DL_Time,Cumulative_YoutubeSess_UMTS_DL_Volume,Max_RSRQ,Max_SNR
  # Cumulative_Full_Service_Time_UMTS,Cumulative_Lim_Service_Time_UMTS,Cumulative_No_Service_Time_UMTS,Cumulative_Full_Service_Time_LTE
  # Cumulative_Lim_Service_Time_LTE,Cumulative_No_Service_Time_LTE,User_Satisfaction

  train_data = train_dataset.loc[:, "Cumulative_YoutubeSess_LTE_DL_Time":"Cumulative_No_Service_Time_LTE"]
  train_target = train_dataset.loc[:, "User_Satisfaction"]
  test_data = test_dataset.loc[:, "Cumulative_YoutubeSess_LTE_DL_Time":"Cumulative_No_Service_Time_LTE"]

  # Pre-processing
  #train_dataset_norm = preprocessing.normalize(train_data)
  X = train_data
  Y = train_target
  validation_size = 0.30
  seed = 7
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

  # Data Plot
  """
  print(test_data.describe())
  print(train_data.describe())
  
  plt.subplot(2, 1, 1)
  plt.plot(train_data)
  plt.subplot(2, 1, 2)
  plt.plot(train_dataset_norm)
  plt.show()"""

  # Select classification model
  dict_classifiers = {
      "Logistic Regression": LogisticRegression(),
      #"Nearest Neighbors": KNeighborsClassifier(),
      #"Linear SVM": SVC(),
      #"Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
      #"Decision Tree": tree.DecisionTreeClassifier(),
      #"Random Forest": RandomForestClassifier(n_estimators=1000),
      #"Neural Net": MLPClassifier(alpha = 1),
      #"Naive Bayes": GaussianNB(),
      #"AdaBoost": AdaBoostClassifier(),
      #"QDA": QuadraticDiscriminantAnalysis(),
      #"Gaussian Process": GaussianProcessClassifier()
  }
  dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8)
  display_dict_models(dict_models)

  # Processing
  for key, val in dict_models.items():
    prediction = val["model"].predict(test_data)
    print(prediction)

  index =  test_dataset.loc[:, "User_Id"]
  prediction = pd.DataFrame(prediction, index= index)
  print(prediction)
  pd.DataFrame(prediction).to_csv("prediction.csv")




