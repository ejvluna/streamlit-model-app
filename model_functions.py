# ... (Import necessary libraries like sklearn, numpy, etc.)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate_gnb(df, features, target, evaluation_method="Train-Test Split"):
    st.subheader('Gaussian Naive Bayes Results')

    X = df[features]  
    y = df[target]

    if evaluation_method == "Train-Test Split":

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Train a Gaussian Naive Bayes classifier
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

        # --- Visualize results ---
        # (Note: Moved this part up to avoid recalculating 'report')
        # Remove 'accuracy', 'macro avg', and 'weighted avg' from the dictionary
        report.pop('accuracy')
        report.pop('macro avg')
        report.pop('weighted avg', None)  # Safely remove if not present

        df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})

        # Create a figure and axes for multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Define the metrics for plotting
        metrics = ['precision', 'recall', 'f1-score']

        # Plot each metric as a separate bar chart
        for i, metric in enumerate(metrics):
            sns.barplot(data=df_report, x='Class', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.capitalize()} by Class')
            axes[i].set_ylabel(f'{metric.capitalize()}')
            axes[i].set_xlabel('Class')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        st.pyplot(fig)
        
    elif evaluation_method == "10-Fold Cross-Validation":
        # ... (You would implement 10-fold cross-validation here)
    else:
        st.error("Invalid evaluation method selected.")

