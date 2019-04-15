import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
  """
  This method, takes as input the X, Y matrices of the Train and Test set.
  And fits them on all of the Classifiers specified in the dict_classifier.
  The trained models, and accuracies are saved in a dictionary.
  """

  dict_models = {}
  for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
    t_start = time.process_time()
    classifier.fit(X_train, Y_train)
    t_end = time.process_time()

    t_diff = t_end - t_start
    train_score = classifier.score(X_train, Y_train)
    test_score = classifier.score(X_test, Y_test)

    dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                    'train_time': t_diff}
    if verbose:
      print("Trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
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

  #TRAIN DATASET
  # User_Id,Cumulative_YoutubeSess_LTE_DL_Time,Cumulative_YoutubeSess_LTE_DL_Volume,Cumulative_YoutubeSess_UMTS_DL_Time,Cumulative_YoutubeSess_UMTS_DL_Volume,Max_RSRQ,Max_SNR
  # Cumulative_Full_Service_Time_UMTS,Cumulative_Lim_Service_Time_UMTS,Cumulative_No_Service_Time_UMTS,Cumulative_Full_Service_Time_LTE
  # Cumulative_Lim_Service_Time_LTE,Cumulative_No_Service_Time_LTE,User_Satisfaction
  test_dataset = pd.read_csv("test_data.csv")
  train_dataset = pd.read_csv("train_data.csv")
  train_data = train_dataset.loc[:, "Cumulative_YoutubeSess_LTE_DL_Time":"Cumulative_No_Service_Time_LTE"]
  train_target = train_dataset.loc[:, "User_Satisfaction"]
  test_data = test_dataset.loc[:, "Cumulative_YoutubeSess_LTE_DL_Time":"Cumulative_No_Service_Time_LTE"]

  # Pre-processing
  X = train_data
  Y = train_target
  validation_size = 0.30
  seed = 7
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

  # Data Plot
  plt.plot(train_data)
  plt.show()

  # Select classification model
  dict_classifiers = {
      "AdaBoost": AdaBoostClassifier(),
      "QDA": QuadraticDiscriminantAnalysis(),
      "Logistic Regression": LogisticRegression(solver="lbfgs"),
      "Nearest Neighbors": KNeighborsClassifier(),
      "Linear SVM": SVC(gamma="scale"),
      "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
      "Decision Tree": tree.DecisionTreeClassifier(),
      "Random Forest": RandomForestClassifier(n_estimators=1000),
      "Neural Net": MLPClassifier(alpha=1),
      "Naive Bayes": GaussianNB()

  }
  dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=8)
  display_dict_models(dict_models)

  # Processing
  index = test_dataset.loc[:, "User_Id"]
  for key, val in dict_models.items():
    prediction = val["model"].predict(test_data)
    prediction = pd.DataFrame(prediction, index= index, columns=["User_Satisfaction"])
    pd.DataFrame(prediction).to_csv(key +"_" + "prediction.csv")




