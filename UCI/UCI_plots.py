import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def feature_importance(numeric_features, clf, categorical_features):
    """Return the most important features of the classifier.
    Works only with predefined classifiers
    """
    # Get feature names after preprocessing
    feature_names = (numeric_features.tolist() + 
                    clf.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(categorical_features).tolist())

    # Get feature importances
    importances = clf.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[-15:]  # Get top 15 features

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 15 Most Important Features")
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()


def conf_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Calculate and print additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

