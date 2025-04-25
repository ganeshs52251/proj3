import pandas as pd
import networkx as nx
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import time

# Load dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets.squeeze()  # ensure it's a 1D array

# Normalize features
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Correlation matrix and graph construction
corr_matrix = X_normalized.corr().abs()
threshold = 0.5
edges = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns
         if i != j and corr_matrix.loc[i, j] > threshold]

G = nx.DiGraph()
G.add_edges_from(edges)

# Apply PageRank
ranks = nx.pagerank(G)
top_features = sorted(ranks, key=ranks.get, reverse=True)[:10]

# Print selected top features
print("Top 10 features selected by PageRank:")
for i, feature in enumerate(top_features, 1):
    print(f"{i}. {feature}")

# Accuracy and time using all features (baseline)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42)

clf_all = GaussianNB()
start_all = time.time()
clf_all.fit(X_train_all, y_train_all)
y_pred_all = clf_all.predict(X_test_all)
end_all = time.time()
accuracy_all = accuracy_score(y_test_all, y_pred_all)
print("Accuracy using all 30 features:", round(accuracy_all * 100, 2), "%")
print("Time using all 30 features:", round(end_all - start_all, 4), "seconds")

# Accuracy and time using PageRank-selected features
X_top = X_normalized[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.3, random_state=42)

clf = GaussianNB()
start_pr = time.time()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_pr = time.time()
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using top 10 PageRank features:", round(accuracy * 100, 2), "%")
print("Time using top 10 features:", round(end_pr - start_pr, 4), "seconds")