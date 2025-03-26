# src/train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os


def load_processed_data(filepath):
    data = pd.read_csv(filepath)
    return data


def train_decision_tree(data):
    # Use an expanded set of features for training
    features = ['ROA', 'Leverage', 'liquidity_ratio', 'coverage_ratio', 'capitalization_ratio']
    X = data[features]
    y = data['classification']

    # Fill missing values with the median of each column
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42, max_depth=6)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved at {filepath}")


def export_tree_visualization(model, feature_names, output_file="tree.dot"):
    export_graphviz(model, out_file=output_file, feature_names=feature_names, class_names=model.classes_, filled=True)
    print(f"Decision tree visualization exported to {output_file}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    data = load_processed_data('data/financials_processed.csv')
    model = train_decision_tree(data)
    save_model(model, "models/decision_tree_model.joblib")
    export_tree_visualization(model, ['ROA', 'Leverage', 'liquidity_ratio', 'coverage_ratio', 'capitalization_ratio'],
                              output_file="models/tree.dot")
