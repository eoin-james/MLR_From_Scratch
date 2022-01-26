import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

cols = ["calorific_value", "nitrogen",
        "turbidity", "style",
        "alcohol", "sugars",
        "bitterness", "beer_id",
        "colour", "degree_of_fermentation"]

file = '/Users/eoinmac/PycharmProjects/MachineLearning/Assignment_2/beer.txt'

data_df = pd.read_csv(file, sep='\t', engine='python', header=None)

cols = ["calorific_value", "nitrogen",
        "turbidity", "style",
        "alcohol", "sugars",
        "bitterness", "beer_id",
        "colour", "degree_of_fermentation"]

data_df.columns = cols  # Add columns to the data, 0...N if not given.

x_data = data_df.iloc[:, data_df.columns != "style"]
y_data = data_df.iloc[:, data_df.columns == "style"]

x_train, x_test,  y_train, y_test = train_test_split(x_data, y_data, train_size=0.7)

lb = LabelEncoder()
y_train = np.asarray(lb.fit_transform(y_train))
y_test = np.asarray(lb.transform(y_test))
# Feature Scaling - Only do to training set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

logreg = LogisticRegression(random_state=1)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix, 'cm')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred), 'report')

print(len(y_test), len(y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
