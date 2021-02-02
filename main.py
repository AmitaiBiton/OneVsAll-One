import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.utils import shuffle
#--------------------------One Vs All -----------------------------------------
def  OneVsAll(X_train, X_test, y_train, y_test ):
    classArray = []
    y_pred = []
    H_theta = []
    maxVal=0
    max = []
    Y = 0
    for i in range(len(y_train)):
        classArray.append(y_train[i][0]) if y_train[i][0] not in classArray else classArray
    newY = np.zeros(len(y_train))
    for i in range(len(X_test)):
        maxVal = 0
        for j in range(len(classArray)):
            newY = np.zeros(len(y_train))
            y = classArray[j]
            for k in range(len(y_train)):
                if y_train[k][0]==y:
                    newY[k] = 1
                elif y_train[j][0]!=y:
                    newY[k] =0
            model = LogisticRegression()
            newY = newY.astype('int')
            H_theta.append(model.fit(X_train, newY))
            maxV = H_theta[j].predict_proba(X_test[i].reshape(1, -1)).flatten()[1]
            if maxVal < maxV:
                maxVal=maxV
                Y= y
        y_pred.append(Y)

    y_pred = np.array(y_pred)
    confusion_matrix = pd.crosstab(y_test.flatten() , y_pred , rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    plt.title("One Vs All")
    plt.show()


def OneVsOne(X_train, X_test, y_train, y_test):
    classArray   =[]
    for i in range(len(y_train)):
        classArray.append(y_train[i][0]) if y_train[i][0] not in classArray else classArray
    newY = np.zeros(len(y_train))
    matrixOneVSOne = np.zeros((6,6))
    """
    for i in range(matrixOneVSOne.shape[0]):
        matrixOneVSOne[i][0] = i
    """
    maxVal =0
    nameClass= 0
    y_pred = []
    print(classArray)
    for i in range(len(X_test)):
        for k  in  range(len(classArray)):
            theOne = classArray[k]
            for j  in range(k+1 , len(classArray)):
                # get only the y you need for two classes
                newDataY = y_train[(y_train == theOne) | (y_train == classArray[j])]
                # newDataY wil be 1 when  in place of theone and zero when other
                newDataY[newDataY == theOne] = 1
                newDataY[newDataY == classArray[j]] = 0
                # get only the X you need for the classes
                X = findXtrain(X_train , y_train , theOne , classArray[j])
                model = LogisticRegression()
                X = np.array(X)
                newDataY = newDataY.astype('int')
                a = model.fit(X , newDataY)
                # insert the predict  in matrix
                matrixOneVSOne[k][j] = a.predict_proba(X_test[i].reshape(1, -1)).flatten()[1]
                if k<6 and j <6 :
                    matrixOneVSOne[j][k] = 1 - a.predict_proba(X_test[i].reshape(1, -1)).flatten()[1]
        maxVal =0
        # get the max value on the matrix by sum of rows
        for l in range(matrixOneVSOne.shape[0]):
            if maxVal < np.sum(matrixOneVSOne[l]):
                maxVal = np.sum(matrixOneVSOne[l])
                nameClass =l
        # append the name of the class that give you the max
        y_pred.append(classArray[nameClass])

    y_pred = np.array(y_pred)
    confusion_matrix = pd.crosstab(y_test.flatten(), y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    plt.title("One Vs One")
    plt.show()




def findXtrain(Xtrain , Ytrain , name , name2):
    new_xtrain = []
    for i in range(len(X_train)):
        if Ytrain[i]==name or Ytrain[i]==name2:
            new_xtrain.append(Xtrain[i])
    return new_xtrain

if  __name__ == "__main__" :
    # -----------------------------------------------------------------------
    # read the excel file to data frame
    excel_data_df = pd.read_excel('BreastTissue (1).xlsx', sheet_name='Data')
    # get x and y
    x = excel_data_df[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP', 'DR', 'P']]
    y = excel_data_df[['Class']]
    y = y.to_numpy()
    # normalized matrix x
    matrix_X = StandardScaler().fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(matrix_X, y, test_size=0.33, random_state=42)

    OneVsAll(X_train, X_test, y_train, y_test )

    OneVsOne(X_train, X_test, y_train, y_test)




