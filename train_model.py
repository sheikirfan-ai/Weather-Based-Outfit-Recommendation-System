# Train + evaluate Decision Tree, save model for Streamlit
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

df = pd.read_csv("data/weather_outfit.csv")

# One-hot encode Occasion
X = pd.get_dummies(df.drop(columns="outfit_category"), columns=["occasion"])
y = df["outfit_category"]
feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Tune depth etc. to avoid overfitting
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {"max_depth": [3, 4, 5, 6, 8, None],
     "min_samples_leaf": [1, 3, 5, 10],
     "criterion": ["gini", "entropy"]},
    cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best params:", grid.best_params_)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"\nTest accuracy: {acc:.3f}\n")
print(classification_report(y_test, pred))

# Plots
ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=model.classes_),
                       display_labels=model.classes_).plot(xticks_rotation=20)
plt.tight_layout(); plt.savefig("plots/confusion_matrix.png", dpi=120); plt.close()

imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
imp.plot(kind="barh", title="Feature Importance"); plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=120); plt.close()

plt.figure(figsize=(22, 10))
plot_tree(model, feature_names=feature_cols, class_names=model.classes_,
          filled=True, max_depth=3, fontsize=8)
plt.savefig("plots/decision_tree.png", dpi=120); plt.close()

# Save model + metadata for the app
report = classification_report(y_test, pred, output_dict=True)
joblib.dump({"model": model, "features": feature_cols,
             "accuracy": acc, "report": report}, "model.pkl")
print("Saved model.pkl")
