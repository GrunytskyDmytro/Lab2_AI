# import numpy as np
# from sklearn import preprocessing
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import f1_score
#
# # Вхідний файл, який містить дані
# input_file = 'income_data.txt'
#
# # Читання даних
# X = []
# y = []
# count_class1 = 0
# count_class2 = 0
# max_datapoints = 25000
#
# with open(input_file, 'r') as f:
#     for line in f.readlines():
#         if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
#             break
#         if '?' in line:
#             continue
#
#         data = line[:-1].split(', ')
#         if data[-1] == '<=50K' and count_class1 < max_datapoints:
#             X.append(data)
#             count_class1 += 1
#         if data[-1] == '>50K' and count_class2 < max_datapoints:
#             X.append(data)
#             count_class2 += 1
#
# # Перетворення на масив numpy
# X = np.array(X)
#
# # Перетворення рядкових даних на числові
# label_encoder = []
# X_encoded = np.empty(X.shape)
# for i, item in enumerate(X[0]):
#     if item.isdigit():
#         X_encoded[:, i] = X[:, i]
#     else:
#         label_encoder.append(preprocessing.LabelEncoder())
#         X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
# X = X_encoded[:, :-1].astype(int)
# y = X_encoded[:, -1].astype(int)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
#
# # 1) Поліноміальне ядро
# poly_classifier = SVC(kernel='poly', degree=8)
# poly_classifier.fit(X_train, y_train)
#
# y_test_pred_poly = poly_classifier.predict(X_test)
#
# f1_poly = cross_val_score(poly_classifier, X, y, scoring='f1_macro', cv=3)
# print("F1 score (Polynomial Kernel): " + str(round(100*f1_poly.mean(), 2)) + "%")
#
# # 2) Гаусове ядро (RBF)
# rbf_classifier = SVC(kernel='rbf')
# rbf_classifier.fit(X_train, y_train)
#
# y_test_pred_rbf = rbf_classifier.predict(X_test)
#
# f1_rbf = cross_val_score(rbf_classifier, X, y, scoring='f1_macro', cv=3)
# print("F1 score (RBF Kernel): " + str(round(100*f1_rbf.mean(), 2)) + "%")
#
# # 3) Сигмоїдальне ядро
# sigmoid_classifier = SVC(kernel='sigmoid')
# sigmoid_classifier.fit(X_train, y_train)
#
# y_test_pred_sigmoid = sigmoid_classifier.predict(X_test)
#
# f1_sigmoid = cross_val_score(sigmoid_classifier, X, y, scoring='f1_macro', cv=3)
# print("F1 score (Sigmoid Kernel): " + str(round(100*f1_sigmoid.mean(), 2)) + "%")
#
# # Порівняння результатів
# print("\nF1 Score Comparison:")
# print("Polynomial Kernel:", round(100*f1_poly.mean(), 2), "%")
# print("RBF Kernel:", round(100*f1_rbf.mean(), 2), "%")
# print("Sigmoid Kernel:", round(100*f1_sigmoid.mean(), 2), "%")
#
# # Оцінка на тестових даних
# f1_test_poly = f1_score(y_test, y_test_pred_poly, average='macro')
# f1_test_rbf = f1_score(y_test, y_test_pred_rbf, average='macro')
# f1_test_sigmoid = f1_score(y_test, y_test_pred_sigmoid, average='macro')
#
# print("\nF1 Score on Test Data:")
# print("Polynomial Kernel:", round(100*f1_test_poly, 2), "%")
# print("RBF Kernel:", round(100*f1_test_rbf, 2), "%")
# print("Sigmoid Kernel:", round(100*f1_test_sigmoid, 2), "%")



import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = SVC(kernel="poly", degree=8)
classifier.fit(X_train_scaled, y_train)
y_test_pred = classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("Accuracy score: {:.2f}%".format(100 * accuracy))
print("Precision score: {:.2f}%".format(100 * precision))
print("Recall score: {:.2f}%".format(100 * recall))
print("F1 score: {:.2f}%".format(100 * f1))
