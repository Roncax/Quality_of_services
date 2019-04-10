import pandas as pd
import platform
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
pd.set_option('display.max_columns', None)

os = platform.system()
if os == "Linux":
    path_train = "/home/roncax/Documents/Git Project/QoS_ML/train_data.csv"
    path_sample = "/home/roncax/Documents/Git Project/QoS_ML/sampleSubmission.csv"
    path_test = "/home/roncax/Documents/Git Project/QoS_ML/test_data.csv"
elif os == "Windows":
    path_train = "A:\Documents\Git Project\QoS_ML\\train_data.csv"
    path_sample = "A:\Documents\Git Project\QoS_ML\sampleSubmission.csv"
    path_test = "A:\Documents\Git Project\QoS_ML\\test_data.csv"

# data = pd.read_csv("/home/roncax/Documents/Git Project/QoS_ML/train_data.csv", nrows=0)

test_dataset = pd.read_csv(path_test)
sample_dataset = pd.read_csv(path_sample)
train_dataset = pd.read_csv(path_train)


print(train_dataset.shape)
print(sample_dataset.shape)
print(test_dataset.shape)

print(type(train_dataset))
x = preprocessing.normalize(train_dataset)

print(type(x))

"""
print(test_dataset.describe())
print(sample_dataset.describe())
print(train_dataset.describe())"""
train_dataset = train_dataset.to_numpy()

plt.subplot(2,1,1)
plt.plot(train_dataset)
plt.title("Not standardized")
plt.subplot(2,1,2)
plt.title("Standardized")
plt.plot(x)
print(x)
#train_dataset.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
#train_dataset.hist()

# scatter_matrix(train_dataset)
plt.show()