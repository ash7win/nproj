from sklearn import tree

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


clf = tree.DecisionTreeClassifier()

clf= clf.fit(X,y)
prediction = clf.predict([[156,74,38],[147,52,40],[190,70,43]])
print (prediction)