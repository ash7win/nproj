#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data= pd.read_csv("D:\Study\programming\DataP7.csv", sep=';')


# In[2]:


data


# In[3]:


data.describe()


# In[5]:


from matplotlib import pyplot as plt
data.hist(bins=10,figsize=(10,10))
plt.show()


# In[6]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(data,test_size=0.2, random_state=73)


# In[7]:


data_train=train_set.copy()


# In[9]:


from pandas.plotting import scatter_matrix
attribs=['BMI','WTKG','Post_BP','MORT20']
scatter_matrix(data_train[attribs],figsize=(12,8))


# In[10]:


corr_matrix=data_train.corr()
corr_matrix['MORT20'].sort_values(ascending=False)


# In[46]:


data_labels=data_train['MORT20'].copy()
data_prepared=data_train.drop(['MORT20','Pre_BP','BMI_Cat','Sex','ID','Alc','SMOKE','BPSTAT','HTM'],axis=1)
data_prepared.shape
data_labels.shape


# In[47]:


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(data_prepared,data_labels)


# In[51]:


some_data_prepared = data_prepared.iloc[:10]
some_labels = data_labels.iloc[:10]
print("Predictions:", log_reg.predict(some_data_prepared))


# In[52]:


print("Labels:", list(some_labels))


# In[53]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_labels)


# In[54]:


some_data_prepared = data_prepared.iloc[:10]
some_labels = data_labels.iloc[:10]
print("Predictions:", tree_reg.predict(some_data_prepared))


# In[55]:


print("Labels:", list(some_labels))


# In[57]:


import numpy as np
from sklearn.metrics import mean_squared_error
data_predictions = tree_reg.predict(data_prepared)
tree_mse = mean_squared_error(data_labels, data_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[58]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, data_prepared, data_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[60]:


def display_them_scores(scores):
    print ('Scores: ', scores)


# In[61]:


display_them_scores(tree_rmse_scores)


# In[62]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(data_prepared, data_labels)


# In[63]:


some_data_prepared = data_prepared.iloc[:10]
some_labels = data_labels.iloc[:10]
print("Predictions:", forest_reg.predict(some_data_prepared))


# In[83]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(data_prepared, data_labels)


# In[77]:


grid_search.best_params_


# In[78]:


grid_search.best_estimator_


# In[79]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[91]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist={
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=4)
}
rnd_search= RandomizedSearchCV(forest_reg, param_distributions=param_dist,
                              n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=73)
rnd_search.fit(data_prepared, data_labels)


# In[92]:


rnd_search.best_params_


# In[93]:


cvres=rnd_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres["params"]):
    print(np.sqrt(-mean_score),params)


# In[98]:


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
X,y=mnist["data"], mnist["target"]    


# In[184]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt 
some_digit = X[3500]

some_digit_image= some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()


# In[100]:


y[3500]


# In[146]:


import numpy as np
shuff= np.random.permutation(6000)
X_train,y_train=X_train[shuff],y_train[shuff]


# In[148]:





# In[103]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)


# In[104]:


from sklearn.linear_model import SGDClassifier

sgd_clf= SGDClassifier(random_state=73)
sgd_clf.fit(X_train, y_train_0)


# In[107]:


sgd_clf.predict([some_digit])


# In[109]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring='accuracy')


# In[111]:


from sklearn.base import BaseEstimator
class Never0Classifier(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
never_0_clf=Never0Classifier()
cross_val_score(never_0_clf,X_train,y_train_0,cv=3,scoring='accuracy')


# In[112]:


from sklearn.model_selection import cross_val_predict
y_train_pred= cross_val_predict(sgd_clf,X_train, y_train_0, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_0,y_train_pred)


# In[113]:


from sklearn.metrics import precision_score,recall_score
precision_score(y_train_0,y_train_pred)


# In[114]:


recall_score(y_train_0,y_train_pred)


# In[115]:


from sklearn.metrics import f1_score
f1_score(y_train_0,y_train_pred)


# In[117]:


y_scores= sgd_clf.decision_function([some_digit])
y_scores


# In[125]:


threshold=0
y_some_digit_pred= (y_scores > threshold)
y_some_digit_pred


# In[126]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3,
                             method="decision_function")


# In[129]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[130]:


y_train_pred_90 = (y_scores > 70000)


# In[131]:


precision_score(y_train_0, y_train_pred_90)


# In[133]:


recall_score(y_train_0, y_train_pred_90)


# In[134]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_0,y_scores)


# In[137]:


from sklearn.ensemble import RandomForestClassifier
forest_clf= RandomForestClassifier(random_state=73)
y_probas_forest=cross_val_predict(forest_clf,X_train,y_train_0,cv=3,method='predict_proba')


# In[141]:


sgd_clf.fit(X_train,y_train)


# In[142]:


sgd_clf.predict([some_digit])


# In[143]:


some_digit_scores= sgd_clf.decision_function([some_digit])
some_digit_scores


# In[151]:


sgd_clf.classes_[1]


# In[158]:


import pandas as pd
data=pd.read_csv('mnist.csv', sep=',')


# In[159]:


data


# In[161]:


y=data['label'].copy()


# In[164]:


X=data.drop(['label'], axis=1)
X


# In[235]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=73)


# In[171]:


y_train.value_counts()


# In[236]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
model.fit(X_train, y_train)


# In[222]:


some_data_prepared = X_train.iloc[:10]
some_labels = y_train.iloc[:10]
print("Predictions:", model.predict(some_data_prepared))


# In[224]:


knn_pred= model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, knn_pred)


# In[237]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[226]:


grid_search.best_params_


# In[233]:


grid_search.best_score_


# In[238]:


from sklearn.metrics import accuracy_score

y_pred= grid_search.predict(X_test)
accuracy_score(y_test, y_pred)


# In[223]:


y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print (some_labels)


# In[203]:


confusion_matrix(y_train_5, y_train)


# In[218]:


from sklearn.model_selection import cross_val_predict
y_train_pred= cross_val_predict(model,X_train, y_train, cv=3)


# In[208]:


f1_score(y_train, y_train_pred,average='macro')


# In[211]:


cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")


# In[210]:


confusion_matrix(y_train, y_train_pred)


# In[219]:


y_train_pred= cross_val_predict(model,X_train, y_train_5, cv=3, method='predict_proba')


# In[ ]:


plt.plot(x=precisions, y= recalls)
plt.show()

