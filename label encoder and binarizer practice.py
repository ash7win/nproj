#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('train.csv')
data=data.dropna()


# In[ ]:


X=data.drop(['Survived','Ticket','Cabin','Name','Embarked'],axis=1)
y=data['Survived'].copy()
X


# In[ ]:


from matplotlib import pyplot as plt
X.hist(bins=50,figsize=(10,10))
plt.show()


# In[ ]:


corr=data.corr()
corr['Survived'].sort_values(ascending=False)


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label=LabelBinarizer()
X_lab= X['Sex']
X['Sex']=label.fit_transform(X_lab)


# In[ ]:


X


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

sgd_clf = DecisionTreeClassifier(random_state=73)
sgd_clf.fit(X,y)


# In[ ]:


import numpy as np
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, X, y,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
tree_rmse_scores


# In[ ]:


some_data_prepared = X.iloc[:10]
some_labels = y.iloc[:10]
print("Predictions:", sgd_clf.predict(some_data_prepared))


# In[ ]:


print("Labels:", list(some_labels))


# In[ ]:


t_data=pd.read_csv('test.csv')


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names
    def fit(self,X,y=None) :
        return self
    def transform (self,X,y=None):
        return X[self._feature_names]


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
num_prep_pipeline= Pipeline([
    ('remover', FeatureRemover(["Age", "SibSp", "Parch", "Fare"])),
    ('imputer', Imputer(strategy='median'))   
])
test_data=num_prep_pipeline.fit_transform(t_data)

cat_pipeline = Pipeline([
        ("select_cat",FeatureRemover(["Pclass", "Sex", "Embarked"])),
        ("cat_encoder", LabelBinarizer()),
    ])
test_data=cat_pipeline.fit_transform(test_data)


# In[ ]:


dana= pd.read_csv("865441c9-498a-4a3f-8f52-3a865c1c421a.csv")
dana
dana.corr()


# In[ ]:


dana


# In[ ]:


print("Predictions:", sgd_clf.predict(test_data))


# In[ ]:





# In[42]:


dana= pd.read_csv("Supply_Chain_Shipment_Pricing_Data.csv")
dana


# In[43]:



from sklearn.preprocessing import LabelBinarizer
l_b=LabelBinarizer()
y = dana['first line designation']
l_b.fit_transform(y)


# In[44]:


import matplotlib.pyplot as plt
dana1.hist(bins=50,figsize=(7,7))
plt.show()


# In[ ]:


dana['shipment mode'].value_counts()


# In[45]:


dana1=dana
dana1


# In[46]:


dana1=pd.get_dummies(dana1,prefix=['mode'],columns=['shipment mode'])
dana1
dana_labels=dana1['line item value']
dana1=dana1.drop('line item value', axis=1)


# In[63]:


corr_m = dana.corr()
corr_m['line item value'].sort_values(ascending=False) 


# In[48]:


from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
class DataFrameSelector( BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names=attributes_names
    def fit(self,X,y=0):
        return self
    def transform(self, X):
        return X[self.attributes_names].values


# In[49]:


class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)


# In[55]:


num_attribs=['line item insurance (usd)', 'line item quantity','mode_Air Charter']    
num_pipeline=Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('std_scaler', StandardScaler()),
        ])
cat_attrib=['first line designation']
cat_pipeline=Pipeline([
    ('selector', DataFrameSelector(cat_attrib)),
    ('binarizer', MyLabelBinarizer())
])


# In[56]:


from sklearn.pipeline import FeatureUnion
full_pipeline= FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])


# In[57]:


data_prepared = full_pipeline.fit_transform(dana1)
data_prepared


# In[58]:


from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(data_prepared, dana_labels)


# In[61]:


some_data = dana1.iloc[5:]
some_labels = dana_labels.iloc[5:]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# In[62]:



print("Labels:", list(some_labels))


# In[68]:


import pandas as pd 
import numpy as np
data= pd.read_csv("2016-first-quarter-nuisance-complaints-.csv")
data


# In[69]:


data=data.dropna()


# In[70]:


data


# In[72]:



from sklearn.preprocessing import LabelBinarizer
l_b=LabelBinarizer()
y = data['Report/No Report']
l_b.fit_transform(y)


# In[ ]:




