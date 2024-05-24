import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold


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


def train_and_evaluate_knn(df, features, target, evaluation_method, k=None):
    st.subheader('k-Nearest Neighbors Results')

    X = df[features]
    y = df[target]

    if evaluation_method == "Train-Test Split":

        if k is None:
            # Find optimal k using a single train-test split
            k_values = range(3, 21, 2)
            accuracies = []
            f1_scores = []
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=1))

            best_k = k_values[np.argmax(accuracies)]
            st.write(f'Best k (n_neighbors): {best_k}')
        else:
            # Use the provided k if given
            best_k = k  # Assign provided k to best_k

        # Split the data (this should be outside the if k is None condition)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train the classifier using the best k
        knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
        knn_classifier.fit(X_train, y_train)

        # Predict and evaluate 
        y_pred = knn_classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Display results
        st.subheader('Model Performance')
        st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
        st.write(f'F1-Score: {f1_score(y_test, y_pred, average="weighted"):.3f}')
        st.text(classification_report(y_test, y_pred))

        #Visualize
        # ... (Your plotting code using report_data remains the same)
        report.pop('accuracy')
        report.pop('macro avg', None)  
        report.pop('weighted avg', None)  
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
        # If k is None, find the best k using cross-validation
        if k is None:
            k_values = range(3, 21, 2)  
            best_k, _ = find_best_k_knn(X, y, k_values, evaluation_method)
            st.write(f'Best k (n_neighbors): {best_k}')
        else:
            best_k = k  # Use the provided k if given
        # Cross-Validation (same for both 'Choose k' and 'Find Optimal k')
        knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='accuracy')

        # Display Results
        st.subheader(f"10-Fold Cross-Validation Results (k = {best_k})")  # Display the best k
        st.write(f"Mean Accuracy: {cv_scores.mean():.3f}")
        st.write(f"Standard Deviation of Accuracy: {cv_scores.std():.3f}")


    else:
        st.error("Invalid evaluation method selected.")




def find_best_k_knn(X, y, k_values, evaluation_method="10-Fold Cross-Validation"):
    if evaluation_method == "10-Fold Cross-Validation":
        mean_accuracies = []
        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            scores = cross_val_score(knn_classifier, X, y, cv=cv, scoring='accuracy')
            mean_accuracies.append(scores.mean())

    elif evaluation_method == "Train-Test Split":
        mean_accuracies = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            y_pred = knn_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mean_accuracies.append(accuracy)

    else:
        raise ValueError("Invalid evaluation method. Choose '10-Fold Cross-Validation' or 'Train-Test Split'.")

    # Find the best k
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)

    return best_k, best_accuracy

