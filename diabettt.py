
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Loading the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)
print(data.info(),data.head())

# Data Preprocessing
scaler = StandardScaler()
X_normalized = scaler.fit_transform(data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = X_normalized

X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# Exploratory Data Analysis (EDA)
print(data.shape)
print(data.head(20))
print(data.describe())

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
dt_f1_score = f1_score(y_test, dt_predictions)
class_report_dt = classification_report(y_test, dt_predictions)

# Decision Tree Model
print("Decision Tree - Accuracy:", dt_accuracy)
print("Decision Tree - MSE:", dt_mse)
print("Decision Tree - RMSE:", dt_rmse)
print("Decision Tree - F1 Score:", dt_f1_score)
print("\nClassification Report - Decision Tree:\n", class_report_dt)

# Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_f1_score = f1_score(y_test, rf_predictions)
class_report_rf = classification_report(y_test, rf_predictions)

# Random Forest Model
print("\nRandom Forest - Accuracy:", rf_accuracy)
print("Random Forest - MSE:", rf_mse)
print("Random Forest - RMSE:", rf_rmse)
print("Random Forest - F1 Score:", rf_f1_score)
print("\nClassification Report - Random Forest:\n", class_report_rf)


# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_f1_score = f1_score(y_test, lr_predictions)
class_report_lr = classification_report(y_test, lr_predictions)

# Logistic Regression Model
print("\nLogistic Regression - Accuracy:", lr_accuracy)
print("Logistic Regression - MSE:", lr_mse)
print("Logistic Regression - RMSE:", lr_rmse)
print("Logistic Regression - F1 Score:", lr_f1_score)
print("\nClassification Report - Logistic Regression:\n", class_report_lr)

# Neural Network Model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=40, validation_data=(X_test, y_test), verbose=1)

#Predictions
nn_predictions_prob = model.predict(X_test)
threshold = 0.5
nn_predictions = np.where(nn_predictions_prob > threshold, 1, 0)

#Accuracy ve MSE 
nn_accuracy = accuracy_score(y_test, nn_predictions)
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
nn_f1_score = f1_score(y_test, nn_predictions)


class_report_nn = classification_report(y_test, nn_predictions)

# Neural Network Model
print("\nNeural Network - Accuracy:", nn_accuracy)
print("Neural Network - MSE:", nn_mse)
print("Neural Network - RMSE:", nn_rmse)
print("Neural Network - F1 Score:", nn_f1_score)
print("\nClassification Report - Neural Network:\n", class_report_nn)

nn_predictions_prob = model.predict(X_test)
threshold = 0.5
nn_predictions = np.where(nn_predictions_prob > threshold, 1, 0)
nn_accuracy = accuracy_score(y_test, nn_predictions)
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
nn_f1_score = f1_score(y_test, nn_predictions)
class_report_nn = classification_report(y_test, nn_predictions)

# Correlation Matrix
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Confusion Matrix
cm_dt = confusion_matrix(y_test, dt_predictions)
cm_rf = confusion_matrix(y_test, rf_predictions)
cm_lr = confusion_matrix(y_test, lr_predictions)
cm_nn = confusion_matrix(y_test, nn_predictions)

# Heatmap for Confusion Matrix
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=axes[0, 0])
axes[0, 0].set_title('Decision Tree Confusion Matrix')
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Confusion Matrix')
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=axes[1, 0])
axes[1, 0].set_title('Logistic Regression Confusion Matrix')
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=axes[1, 1])
axes[1, 1].set_title('Neural Network Confusion Matrix')
plt.show()

# Decision Tree 
dt_model = DecisionTreeClassifier(max_depth=4)
dt_model.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
plt.show()


print("\nConfusion Matrix - Decision Tree:\n", confusion_matrix(y_test, dt_predictions))


print("\nConfusion Matrix - Random Forest:\n", confusion_matrix(y_test, rf_predictions))


print("\nConfusion Matrix - Logistic Regression:\n", confusion_matrix(y_test, lr_predictions))

print("\nConfusion Matrix - Neural Network:\n", confusion_matrix(y_test, nn_predictions))



# Modellerin İsimleri
models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Neural Network']

# Accuracy Değerleri
accuracy_values = [dt_accuracy, rf_accuracy, lr_accuracy, nn_accuracy]

# MSE Değerleri
mse_values = [dt_mse, rf_mse, lr_mse, nn_mse]

# RMSE Değerleri
rmse_values = [dt_rmse, rf_rmse, lr_rmse, nn_rmse]

# F1 Score Değerleri
f1_score_values = [dt_f1_score, rf_f1_score, lr_f1_score, nn_f1_score]


barWidth = 0.2
r1 = np.arange(len(accuracy_values))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]


plt.figure(figsize=(12, 8))
plt.bar(r1, accuracy_values, color='blue', width=barWidth, edgecolor='grey', label='Accuracy')
plt.bar(r2, mse_values, color='green', width=barWidth, edgecolor='grey', label='MSE')
plt.bar(r3, rmse_values, color='orange', width=barWidth, edgecolor='grey', label='RMSE')
plt.bar(r4, f1_score_values, color='red', width=barWidth, edgecolor='grey', label='F1 Score')


plt.xlabel('Modeller', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(accuracy_values))], models)
plt.legend()
plt.show()