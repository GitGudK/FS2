import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=5000, n_features=1000, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2,
                           random_state=0, shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(20): #range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
#X.shape
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
#clf.feature_importances_
#model = SelectFromModel(clf, prefit=True)
model = SelectFromModel(clf, threshold=0.1, prefit=True)
X_single = model.transform(X)



import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=5000, n_features=1000, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2,
                           random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Build a forest using all features and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X_train, y_train)
# Calculate model performance based on all features
y_pred, y_prob = forest.predict(X_test), forest.predict_proba(X_test)
ScoreAll = roc_auc_score(y_test, y_prob[:,1])

# Select certain features based on ExtraTrees importances
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
model = SelectFromModel(forest, prefit=True)
X_test_single = model.transform(X)

# Calculate model performance based on subset of features from single run
y_pred, y_prob = forest.predict(X_test), forest.predict_proba(X_test)
roc_auc_score(y_test, y_prob[:,1])


# Set the initial global ratings
global_ratings = 1000*np.ones(X.shape[1])

X_mask = model._get_support_mask()
global_ratings[X_mask] += 1
global_ratings[~X_mask] -= 1


