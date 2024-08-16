import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from progress.bar import Bar
from progress.spinner import Spinner

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 
    'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
    'native-country', 'income'
]

spinner = Spinner('Loading dataset...')
data = pd.read_csv('./adult.csv', header=None, names=columns)
spinner.finish()

data = data.dropna()

bar = Bar('Encoding categorical variables', max=len(data.select_dtypes(include=['object']).columns))
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
    bar.next()
bar.finish()

X = data.drop('income', axis=1)
y = data['income']

scaler = StandardScaler()
X = scaler.fit_transform(X)

plot_names = ['dataset_statistics.png', 'income_distribution.png', 'income_pairplot.png', 'correlation_heatmap.png']
bar = Bar('Saving plots', max=len(plot_names))

fig, ax = plt.subplots(figsize=(8, 6))
data.describe().plot(kind='bar', ax=ax)
ax.set_title('Dataset Statistics')
fig.savefig('dataset_statistics.png', dpi=300)  
plt.close(fig)
bar.next()

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='income', data=data, ax=ax)
ax.set_title('Distribution of Income Categories')
fig.savefig('income_distribution.png', dpi=300)  
plt.close(fig)
bar.next()

fig = sns.pairplot(data, hue='income')
fig.fig.suptitle('Pairplot of Features Colored by Income', y=1.02)
fig.savefig('income_pairplot.png', dpi=300)  
plt.close(fig.fig)
bar.next()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
fig.savefig('correlation_heatmap.png', dpi=300)  
plt.close(fig)
bar.finish()

spinner = Spinner('Splitting data into training and testing sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
spinner.finish()


spinner = Spinner('Training Logistic Regression model...')
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
spinner.finish()

spinner = Spinner('Training Decision Tree model...')
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
spinner.finish()

spinner = Spinner('Training Random Forest model...')
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
spinner.finish()


spinner = Spinner('Comparing models...')
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, y_pred_logistic),
                 accuracy_score(y_test, y_pred_tree),
                 accuracy_score(y_test, y_pred_forest)],
    'Precision': [precision_score(y_test, y_pred_logistic),
                  precision_score(y_test, y_pred_tree),
                  precision_score(y_test, y_pred_forest)],
    'Recall': [recall_score(y_test, y_pred_logistic),
               recall_score(y_test, y_pred_tree),
               recall_score(y_test, y_pred_forest)],
    'F1 Score': [f1_score(y_test, y_pred_logistic),
                 f1_score(y_test, y_pred_tree),
                 f1_score(y_test, y_pred_forest)]
})
spinner.finish()

print("\nModel Comparison:")
print(results)

