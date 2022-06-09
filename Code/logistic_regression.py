import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.stdout = open("vif_heart.txt", "w")
data0 = pd.read_csv("./datasets/Heart_Disease_Prediction.csv")

data1 = pd.DataFrame(data0)

# one hot encoding and creating dummy variables
data1["Heart Disease"] = data1['Heart Disease'].map({'Presence': 1, 'Absence': 0})


#print(data1.to_string())
X = data1.drop(["Heart Disease"], axis=1)
y = data1["Heart Disease"]

# introduce a confusion matrix


X_train, X_text, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=10)

logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())
#print(logm1.fit().summary())
x = data1[['Age', 'Sex', 'Chest pain type', 'BP',
          'Cholesterol', 'FBS over 120','EKG results',
          'Max HR', 'Exercise angina', 'ST depression',
          'Slope of ST', 'Number of vessels fluro',
          'Thallium', 'Heart Disease']]
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns

vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                   for i in range(len(x.columns))]
print(vif_data)



#plt.figure(figsize=(20, 10))
#sns.heatmap(data1.columns, annot=True)
#plt.show()
#print(data1.corr().to_string())
sys.stdout.close()
# print(data1.to_string())


