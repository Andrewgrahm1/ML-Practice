from ucimlrepo import fetch_ucirepo
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
The dataset I used in the following algorithm is from the UC Irving Machine Learning Repository
The following is the citation to the dataset:
Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
"""

# confusionMatrix function to create the confusion matrix of the true and predicted values
def confusionMatrix(numLabels, strLabels, Ytest, predKNN):

    # Initial matrix with all 0 for each index and column
    matrixKNN = pd.DataFrame(0, index=numLabels, columns=numLabels)

    # enact for loop to add 1 to each True Positive or False positive combo in matrix
    for i in range(len(predKNN)):
        matrixKNN.loc[Ytest[i], predKNN[i]] += 1

    # Create a figure using the confusion matrix to see the results of the ML tool
    plt.figure(figsize=(16, 8))

    # use a heatmap to display the confusion matrix
    sns.heatmap(matrixKNN, annot=True, fmt='d', cmap='Blues', xticklabels=strLabels, yticklabels=strLabels)

    # assign the labels of the matrix as predicted or true
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # add a title
    plt.title('K-Nearest Neighbors Confusion Matrix')

    # show it
    plt.show()

    # Save figure to directory
    plt.savefig(f"./KNNConfusionMatrix.png")

    # return matrix
    return matrixKNN

# function to find the accuracy of predictions
def accuracy(predictions, Ytest):

    # find the summation of predictions equaling true values and divide by the length of true values
    accuracy = np.sum(predictions == Ytest) / len(Ytest)

    percentage = round(accuracy * 100, 2)

    return f"{percentage}%"

class KNN:

    # Initialization Function with number of K values declared
    def __init__(self, k=3):
        self.k = k

    # fit function to fit xTrain and yTrain values to model
    def fit(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain

    # predict function to run model to predict for xTest Values
    def predict(self, xTest):
        return np.array([self._helpPredict(xVal) for xVal in xTest])

    def _helpPredict(self, xTestVal):
        # compute Euclidean Distances
        distances = [self._euclideanDistances(xTestVal, xTrainVal) for xTrainVal in self.xTrain]

        # Retrieve closest K points
        # Retrieve the indexes of the closest K points
        kIndexes = np.argsort(distances)[:self.k]

        # Retrieve the labels of the closest K points
        kLabels = [self.yTrain[index] for index in kIndexes]

        # Predict Classification and return first value of tuple
        return Counter(kLabels).most_common()[0][0]

    # Helper function to calculate Euclidean Distance between 2 points
    def _euclideanDistances(self, x1, x2):
        return np.sqrt(np.sum(x1-x2)**2)

# # fetch iris dataset (Warning for the Iris dataset I recieved multiple connection problems with this)
# iris = fetch_ucirepo(id=53) 
  
# # data (as pandas dataframes) 
# X = iris.data.features 
# y = iris.data.targets 
# headers = iris.data.headers

# Alternative reading in iris dataset by downloading it
file = './iris/iris.data'

# Define the column names for the Iris dataset
headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Read the CSV file into a DataFrame
df = pd.read_csv(file, header=None, names=headers)

# Separate features and target
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# make a new dataframe using X and columns headers and append y
features = pd.DataFrame(data=X, columns=headers[:4])
targets = y
classLabels = np.unique(targets)

# To be used to view the dataset
print(f"The features of the dataset are:\n{features}")
print(f"\nThe targets of the dataset are:\n{classLabels}")

# Preprocess Data to be used in the model
# initialize data list
data = []

# extract index and row from features
for index, row in features.iterrows():

    # convert to numpy array
    array = row.to_numpy()

    # append numpy array to list
    data.append(array)

# assign features to numpy array of list
features = np.array(data)

# make empty list to store labels
labels = []

# iterate through each value in targets and change to numerical class
for value in targets:
    if value == 'Iris-setosa':
        labels.append(0)
    elif value == 'Iris-versicolor':
        labels.append(1)
    else:
        labels.append(2)

# turn labels list into numpy array
targets = np.array(labels)

# Use scikit-learn function called train_test_split to split the features and targets into X and Y train and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, targets, test_size=0.2, random_state=42)

# Initialize the KNN model
KNearest = KNN(k=5)

# Fit the model to the training data
KNearest.fit(Xtrain, Ytrain)

# Make a prediction with the testing X
predKNN = KNearest.predict(Xtest)

# find matrix for KNN
matrixKNN = confusionMatrix(numLabels=np.unique(targets), strLabels=classLabels, Ytest=Ytest, predKNN=predKNN)

# Print accuracy of run
print(f"\nThis K-Nearest Neighbors model possesses an accuracy of {accuracy(predKNN, Ytest)}")