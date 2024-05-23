# model_functions.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

def train_and_evaluate_gnb(df, features, target, evaluation_method="Train-Test Split"):
    st.subheader('Gaussian Naive Bayes Results')

    X = df[features]
    y = df[target]

    if evaluation_method == "Train-Test Split":
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Train the classifier
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)
        
        # Predict and Evaluate
        y_pred = nb_classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Display Results
        st.subheader('Model Performance')
        st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
        st.write(f'F1-Score: {f1_score(y_test, y_pred, average="weighted"):.3f}')  # Add F1-score
        st.text(classification_report(y_test, y_pred))  # Display full report
        
        # Visualize Results
        report.pop('accuracy')  # Remove overall accuracy (it's already displayed above)
        report.pop('macro avg', None)  # Remove macro average if it exists
        report.pop('weighted avg', None)  # Remove weighted average if it exists

        df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['precision', 'recall', 'f1-score']

        for i, metric in enumerate(metrics):
            sns.barplot(data=df_report, x='Class', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.capitalize()} by Class')
            axes[i].set_ylabel(f'{metric.capitalize()}')
            axes[i].set_xlabel('Class')

        plt.tight_layout()
        st.pyplot(fig)


    elif evaluation_method == "10-Fold Cross-Validation":
        # K-fold Cross Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(GaussianNB(), X, y, cv=kf, scoring="accuracy")

        # Display Results
        st.subheader("10-Fold Cross-Validation Results")
        st.write(f"Mean Accuracy: {cv_scores.mean():.3f}")
        st.write(f"Standard Deviation: {cv_scores.std():.3f}")

    else:
        st.error("Invalid evaluation method selected.")
