#Load libraries
from sklearn.datasets import load_iris #Load dataset
from sklearn.model_selection import train_test_split #Split data
from sklearn.tree import DecisionTreeClassifier #Choose model
from sklearn.metrics import accuracy_score #Evaluate model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Evaluate model, display confusion matrix

import os #lets python work with files and folders
import matplotlib.pyplot as plt #to allow python to make and save plots
import argparse #to convert to pure python command line script
import joblib #to export trained model


def main(test_size, random_state):
    #Set output directory
    folder = "output" #name of folder
    os.makedirs(folder, exist_ok = True) #changes directory to folder and creates it if not exist

    #Data Preparation
    iris = load_iris() #assign to variable
    X = iris.data #data contains measurements - shape of data (150 rows, 4 columns)
    y = iris.target #labels contains the correct answer for each row - shape of data (150 rows, )

    print(iris.feature_names, iris.target_names) #prints the feature (in this case measurements) names and target (correct values) 

    #Split data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state) #random_state = 42 means the shuffle is deterministic so results can be reproduced

    #Choosing and training a model
    model = DecisionTreeClassifier(random_state = 42) 

    #Train the model
    model.fit(X_train, y_train)

    #Make predictions
    y_pred = model.predict(X_test)

    print(f"Predictions: {y_pred[:5]}") #initial inspections
    print(f"True labels: {y_test[:5]}")

    #Evaluate model
    accuracy = accuracy_score(y_test, y_pred) #accuracy score
    print(f"Accuracy: {accuracy}")

    #Test and output model to output directory
    if(accuracy > 0.9):
        joblib.dump(model, os.path.join(folder, "model.joblib"))


    #create confusion matrix
    cm = confusion_matrix(y_test, y_pred) #confusion matrix
    cm_display = ConfusionMatrixDisplay(cm) #creates display object

    cm_display.plot() #plots the confusion matrix
    plt.savefig(os.path.join(folder, "confusion_matrix.png")) #save CURRENT plot
    plt.close()









#EXPLANATION OF RESULTS
"""The detailed notes of what each line of code does are in the iris_model.ipynb
notebook.

-> The results show that the accuracy is 100% which is great.
-> However due to the small dataset this may due to overfitting.

-> To further test the model a larger dataset, bigger test/train split or 
comparison to other models is needed.
-> Additionally, the iterating and improving techniques could be applied if the 
accuracy falls below 100% during further testing of the model.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type = float, default = 0.2)
    parser.add_argument("--random-state", type = int, default = 42)
    args = parser.parse_args()

    main(args.test_size, args.random_state)