import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset
data = {
    'Age': [20, 25, 30, 35, 40, 45, 50],
    'Time_Spent': [10, 20, 30, 40, 50, 60, 70],
    'Interest_Score': [3, 6, 7, 8, 9, 5, 2],
    'Ad_Duration': [5, 10, 15, 20, 25, 30, 35],
    'Clicked_Ad': [0, 1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[['Age', 'Time_Spent', 'Interest_Score', 'Ad_Duration']]
y = df['Clicked_Ad']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model trained and saved as model.pkl")
