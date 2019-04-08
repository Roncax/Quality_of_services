import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path_train = "A:\Documents\Git Project\QoS_ML\\train_data.csv"
path_sample = "A:\Documents\Git Project\QoS_ML\sampleSubmission.csv"
path_test = "A:\Documents\Git Project\QoS_ML\\test_data.csv"

names = []
test_dataset = pandas.read_csv(path_test)
sample_dataset = pandas.read_csv(path_sample)
train_dataset = pandas.read_csv(path_train)

print(train_dataset)