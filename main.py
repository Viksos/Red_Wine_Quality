import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("winequality-red.csv",sep = ",")
#df = df.drop_duplicates()
print(df.isna().sum())

def sprawdzajaca(df):
    for i in df.columns :
        if(df[i].dtypes == 'int64' or df[i].dtypes == 'float64'):
            pass
        else:
            print(df[i][0:10])
sprawdzajaca(df)


for i in range(len(df.columns)):
    print("dla kolumny ",i,'mamy \n',df.iloc[:,i].describe())

for i in ['min','max','mean']:
    df.describe().loc[i].plot(kind = 'bar')
    plt.title(i)
    plt.savefig(i+'.jpg')
    plt.show()


for i in df.columns:
    df[i].value_counts().head().plot(kind = 'bar')
    plt.title(i)
    plt.savefig(i+'.jpg')
    plt.show()
 
from sklearn.model_selection import train_test_split
    
X, y = df.drop(columns=['quality']).to_numpy(),df['quality'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.33, random_state=1)

import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import seaborn as sn

acc = []
xl = ['3','4','5','6','7','8']
################################################################################ drzewo ################################

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
#tree.plot_tree(drzewo) 
print('Dokładnoć rozpoznawania jakosci wina za pomoca drzewa decyzyjnego: ',round(clf.score(X_test,y_test),2)*100,'%')
acc.append(clf.score(X_test,y_test))
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('Drzewo - Macierz błędu')
plt.savefig('Drzewo - Macierz błędu')
plt.show()

################################################################################# Naive Bayes ############################

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train) 
print('Dokładnoć rozpoznawania jakosci wina za pomoca metody Naive Bayes: ',round(clf.score(X_test,y_test),2)*100,'%')
acc.append(clf.score(X_test,y_test))
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('Naive Bayes - Macierz błędu')
plt.savefig('Naive Bayes - Macierz błędu')
plt.show()

#################################################################################### K neighbors #################################

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3 )
kn.fit(X_train,y_train)
y_pred = kn.predict(X_test)
print("Wynik z sąsiadów: ",round(kn.score(X_test,y_test)*100,2),'%')
acc.append(clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('K sąsiadów - Macierz błędu')
plt.savefig('K sąsiadów - Macierz błędu')
plt.show()

###################################################################################### Random Forest #############################

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Wyniki z random forest: ",round(clf.score(X_test,y_test)*100,2),'%')
acc.append(clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('Random Forest - Macierz błędu')
plt.savefig('Random Forest - Macierz błędu')
plt.show()

####################################################################################### SVC #######################################

from sklearn.svm import SVC

clf = SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Wyniki z SVC: ",round(clf.score(X_test,y_test)*100,2),'%')
acc.append(clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('SVM - Macierz błędu')
plt.savefig('SVM - Macierz błędu')
plt.show()

###################################################################################### ACC ###########################################


df_acc = pd.DataFrame({'Metoda' : ['Tree','Naive Bayes','K neighbors','Random Forest','SVM'],'acc' : acc})
df_acc.plot(x = 'Metoda',y = 'acc',kind = 'bar')
plt.savefig('ACC')
plt.show()

################################################################################### Poprawka #################
val = []
for i in df['quality'].value_counts().to_list():
    val.append((i/df['quality'].count())**(-1))

clf = RandomForestClassifier(n_estimators=10,class_weight = {3: val[-1], 4: val[-3], 5: val[-6], 6: val[-5], 7:val[-4], 8:val[-2]})
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Wyniki z random forest: ",round(clf.score(X_test,y_test)*100,2),'%')
acc.append(clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('Random Forest z wagami - Macierz błędu')
plt.savefig('Random Forest z wag - Macierz błędu')
plt.show()

#################################################################################################### Zmiejszanie 
import random
del_5 = random.choices(df[df.quality == 5].index.to_list(),k=int(0.8*len(df[df.quality == 5].index.to_list())))
del_6 = random.choices(df[df.quality == 6].index.to_list(),k=int(0.8*len(df[df.quality == 6].index.to_list())))
df = df.drop(del_6)
df = df.drop(del_5)


X, y = df.drop(columns=['quality']).to_numpy(),df['quality'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Wyniki z random forest: ",round(clf.score(X_test,y_test)*100,2),'%')
acc.append(clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('Random Forest - Macierz błędu')
plt.savefig('Random Forest zmniej - Macierz błędu')
plt.show()


val = []
for i in df['quality'].value_counts().to_list():
    val.append((i/df['quality'].count())**(-1))

clf = RandomForestClassifier(n_estimators=10,class_weight = {3: val[-1], 4: val[-3], 5: val[-6], 6: val[-5], 7:val[-4], 8:val[-2]})
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Wyniki z random forest: ",round(clf.score(X_test,y_test)*100,2),'%')
acc.append(clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

sn.heatmap(confusion_matrix(y_test, y_pred),xticklabels = xl , yticklabels = xl , annot=True)
plt.title('Random Forest z wagami - Macierz błędu')
plt.savefig('Random Forest z wag  zmniej- Macierz błędu')
plt.show()

