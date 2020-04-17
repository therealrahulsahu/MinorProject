import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_UNDEFINED
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn import model_selection
#from sklearn.tree import export_graphviz
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Training.csv')
#testing  = pd.read_csv('Testing.csv')
cols     = training.columns[:-1]
x        = training[cols]
y        = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

print(le.transform(['Acne']))

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
'''
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)
'''
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x,y)
#clf = clf1.fit(x_train,y_train)
'''
print(clf.score(x_train,y_train))
print ("cross result========")
scores = model_selection.cross_val_score(clf, x_test, y_test, cv=3)
print (scores)
print (scores.mean())
print(clf.score(testx,testy))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

#feature_importances
for f in range(10):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

print("Please reply Yes or No for the following symptoms")
'''
def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree_, feature_names):
    feature_name = [
        feature_names[i] if i != TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []
    def recurse(node, depth):

        #indent = "  " * depth
        if tree_.feature[node] != TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(name + " ?")
            ans = input()
            ans = ans.lower()
            if ans == 'yes':
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            for i in symptoms_given:
                print(i +"*?")
                af= input()
                af=af.lower()
                if af=='yes':
                    symptoms_present.append(i)
            print( "You may have " +  present_disease )
            print("symptoms given "  +  str(list(symptoms_given)) )
            print("symptoms present  " + str(list(symptoms_present)))
            confidence_level = (1.0*len(symptoms_present)/len(symptoms_given))
            print("confidence level is " + str(confidence_level))
    recurse(0, 1)
tree_to_code(clf.tree_, cols)

'''
feature_name = [
        cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in clf.tree_.feature
    ]
'''