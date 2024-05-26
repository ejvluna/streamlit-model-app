# Import required libraries
import streamlit as st # for the web app interface on Streamlit
import pandas as pd # for data manipulation
import numpy as np # for numerical operations
from sklearn.preprocessing import StandardScaler # for standardization
from sklearn.preprocessing import LabelEncoder # for encoding the target variable
from sklearn.model_selection import train_test_split # for splitting the data into training and testing sets
from sklearn.naive_bayes import GaussianNB # for Gaussian Naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier # for k-Nearest Neighbors classifier
from sklearn.model_selection import cross_val_score, KFold # for cross-validation
from sklearn.metrics import classification_report, accuracy_score, f1_score # for model evaluation
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for advanced plotting

# Function 1: a function to train and evaluate a Gaussian Naive Bayes classifier 
def train_and_evaluate_gnb(df, features, target, evaluation_method, model_results, model_counter):

    # Display the classifier type
    #st.subheader('Gaussian Naive Bayes Results')
    # Store the model name
    model_name = f"Model {model_counter}"

    # === Phase 1: Data Preprocessing (i.e. missing values, encoding, standardization)===

    # a. Missing Values: Drop rows with missing values from the ENTIRE dataset
    df_clean = df.dropna(subset=features + [target])
    # Create X and y from the clean dataframe
    X = df_clean[features]
    y = df_clean[target]

    # b. Label Encoding: use LabelEncoder to encode the target variable to numerical form
    y = LabelEncoder().fit_transform(y) 

    # c. Standardization: No standardization required for Naive Bayes classifiers since they assume normal distribution
    
    # === Phase 2: Model Training and Evaluation ===

    # Option 1: if user selects "Train-Test Split"
    if evaluation_method == "Train-Test Split":
        # Split the data into training and testing sets of 80% and 20% respectively
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # Create a Gaussian Naive Bayes classifier
        nb_classifier = GaussianNB()
        # Fit the classifier to the training data
        nb_classifier.fit(X_train, y_train)
        # Predict the target variable for the test data
        y_pred = nb_classifier.predict(X_test)
        # Generate classification report to evaluate the model (optional)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Store the performance metrics for the model  in a dictionary with the model name as the key
        model_results[model_name] = {
            'Model Type': 'Gaussian Naive Bayes',
            'Evaluation Method': evaluation_method,
            'Target' : target,
            'k-Value': 0,  # Not applicable for GNB
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            'Features': features
        }

        # Display Performance Metrics using values from the dictionary
        st.subheader(f"{model_name} Performance")
        st.write(f"Model Type: {model_results[model_name]['Model Type']}")
        st.write(f"Evaluation Method: {model_results[model_name]['Evaluation Method']}")
        st.write(f"Accuracy: {model_results[model_name]['Accuracy']:.3f}")
        st.write(f"F1-Score: {model_results[model_name]['F1-Score']:.3f}")
        st.write(f"Target Variable: {target}") 
        st.write("Features Used:")
        for feature in model_results[model_name]['Features']:
            st.write(f"    - {feature}")


    # Option 2: if user selects "10-Fold Cross-Validation"
    elif evaluation_method == "10-Fold Cross-Validation":
        # Create a 10-fold cross-validation object
        kf = KFold(n_splits=10, shuffle=True, random_state=42) 
        # Perform 10-fold cross-validation on the classifier and store the accuracy scores
        cv_scores = cross_val_score(GaussianNB(), X, y, cv=kf, scoring="accuracy") 

        # Store the performance metrics for the model  in a dictionary
        model_results[model_name] = {
            'Model Type': 'Gaussian Naive Bayes',
            'Evaluation Method': evaluation_method,
            'Target' : target,
            'k-Value': 0,  # Not applicable for GNB
            'Accuracy': cv_scores.mean(), 
            'F1-Score': "N/A",  # Not applicable for cross-validation on accuracy
            'Features': features
        }

        # Display Performance Metrics using values from the dictionary
        st.subheader(f"{model_name} Performance")
        st.write(f"Model Type: {model_results[model_name]['Model Type']}")
        st.write(f"Evaluation Method: {model_results[model_name]['Evaluation Method']}")
        st.write(f"Mean Accuracy: {model_results[model_name]['Accuracy']:.3f}")
        st.write(f"Standard Deviation: {cv_scores.std():.3f}") 
        st.write(f"F1-Score: {model_results[model_name]['F1-Score']}")  # No formatting for "N/A"
        st.write(f"Target Variable: {model_results[model_name]['Target']}") 
        st.write("Features Used:")
        for feature in model_results[model_name]['Features']:
            st.write(f"    - {feature}")


    else:
        # Display an error message if the evaluation method is invalid
        st.error("Invalid evaluation method selected.")

    # Return the dictionary with the model results and the updated model counter
    return model_results, model_counter + 1  


# Function 2: a function to train and evaluate a k-Nearest Neighbors classifier
def train_and_evaluate_knn(df, features, target, evaluation_method, model_results, model_counter, k=None):
    
    # Store the model name
    model_name = f"Model {model_counter}" 

    # === Phase 1: Data Preprocessing ===

    # Create a subset of the dataframe with only the selected features and target variable and drop rows with missing values
    df_clean = df.dropna(subset=features + [target])
    # Create X and y from the clean dataframe
    X = df_clean[features]
    y = df_clean[target]
    # Create a label encoder object and encode the target variable to numerical form
    y = LabelEncoder().fit_transform(y)
    # Create a StandardScaler object to standardize the features
    scaler = StandardScaler()
    # Fit and transform the features using the scaler
    X_scaled = scaler.fit_transform(X)  

    # Check if the user has provided a k value
    if k is None:
        # Set the range of k values to test: odd numbers from 3 to 21
        k_values = range(3, 21, 2)
        # Call a helper function to find the best k value
        best_k, _ = find_best_k_knn(X_scaled, y, k_values, evaluation_method)
        # Display the best k value
        st.write(f'Best k (n_neighbors): {best_k}')
    else:
        # Use the provided k value
        best_k = k

    # === Phase 2: Model Training and Evaluation === 

    # Create a k-Nearest Neighbors classifier with the best k value
    knn_classifier = KNeighborsClassifier(n_neighbors=best_k) 

    # Option 1: User selects "Train-Test Split"
    if evaluation_method == "Train-Test Split":
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        # Fit the k-Nearest Neighbors classifier to the training data
        knn_classifier.fit(X_train, y_train)
        # Predict the target variable for the test data using the trained model
        y_pred = knn_classifier.predict(X_test)
        # Generate a classification report to evaluate the model (optional)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Option 2: User selects "10-Fold Cross-Validation"
    elif evaluation_method == "10-Fold Cross-Validation":
        # Create a k-fold cross-validation object and set the number of splits to 10
        kf = KFold(n_splits=10, shuffle=True, random_state=42) # shuffle to ensure randomness, random_state for reproducibility
        # Perform 10-fold cross-validation on the k-Nearest Neighbors classifier and store the accuracy scores
        cv_scores = cross_val_score(knn_classifier, X_scaled, y, cv=kf, scoring='accuracy')
        f1_scores = cross_val_score(knn_classifier, X_scaled, y, cv=kf, scoring='f1_weighted')
    else:
        # Display an error message if the evaluation method is invalid
        st.error("Invalid evaluation method selected.")
    
    # === Phase 3: Display Performance Metrics ===

    # Store the model results in a dictionary with the model name as the key
    model_results[model_name] = {
        'Model Type': 'kNN',
        'Evaluation Method': evaluation_method,
        'Target' : target,
        'k-Value': best_k,
        'Accuracy': accuracy_score(y_test, y_pred) if evaluation_method == "Train-Test Split" else cv_scores.mean(),
        'F1-Score': f1_score(y_test, y_pred, average='weighted') if evaluation_method == "Train-Test Split" else f1_scores.mean(), 
        'Features': features  
    }

    # Display Performance Metrics using values from the dictionary
    st.subheader(f"{model_name} Performance")
    st.write(f"Model Type: {model_results[model_name]['Model Type']}")
    st.write(f"k-Value: {model_results[model_name]['k-Value']}")
    st.write(f"Evaluation Method: {model_results[model_name]['Evaluation Method']}")
    st.write(f"Accuracy: {model_results[model_name]['Accuracy']:.3f}")
    # Conditional Display of Standard Deviation of Accuracy 
    if evaluation_method == "10-Fold Cross-Validation":
        st.write(f"Standard Deviation of Accuracy: {cv_scores.std():.3f}")
    st.write(f"F1-Score: {model_results[model_name]['F1-Score']:.3f}") 
    st.write(f"Target Variable: {model_results[model_name]['Target']}")
    st.write("Features Used:")
    for feature in model_results[model_name]['Features']:
        st.write(f"    - {feature}")

    # Return the dictionary with the model results and the updated model counter
    return model_results, model_counter + 1


# Function 3: a helper function to find the best k for a k-Nearest Neighbors classifier
def find_best_k_knn(X, y, k_values, evaluation_method="10-Fold Cross-Validation"):
    # Create an empty list to store the mean accuracies for each k value
    mean_accuracies = []
    # Option 1: User selects "10-Fold Cross-Validation"
    if evaluation_method == "10-Fold Cross-Validation":
        # Iterate over each k value, and for each do the following:
        for k in k_values:
            # Create a k-Nearest Neighbors classifier with the current k value
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            # Perform 10-fold cross-validation and store the accuracy scores
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            # Calculate the mean accuracy of the classifier
            scores = cross_val_score(knn_classifier, X, y, cv=cv, scoring='accuracy')
            # Add the mean accuracy score to the list
            mean_accuracies.append(scores.mean())

    # Option 2: User selects "Train-Test Split"
    elif evaluation_method == "Train-Test Split":
        # Iterate over each k value, and for each do the following:
        for k in k_values:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Create a k-Nearest Neighbors classifier with the current k value
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            # Fit the classifier to the training data
            knn_classifier.fit(X_train, y_train)
            # Predict the target variable for the test data using the trained model
            y_pred = knn_classifier.predict(X_test)
            # Calculate the accuracy score of the classifier and store it
            accuracy = accuracy_score(y_test, y_pred)
            # Add the accuracy score to the list
            mean_accuracies.append(accuracy)

    else:
        # Raise an error if the evaluation method is invalid
        raise ValueError("Invalid evaluation method. Choose '10-Fold Cross-Validation' or 'Train-Test Split'.")

    # Calculate the best k by comparing the mean accuracy of each k and selecting the one with the highest accuracy
    best_k = k_values[np.argmax(mean_accuracies)]
    # Return the best k value found
    return best_k

# Function 4: a function to display and compare model results in a Streamlit app
def display_and_compare_models(model_results):
    """
    Displays and compares model results in a Streamlit app.

    Args:
        model_results (dict): A dictionary containing model results, where keys are model names and values are dictionaries of metrics.
    """

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(model_results).transpose()

    # Set Pandas display options to prevent truncation
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # Format Accuracy and F1-Score to four decimal places if the value is numeric
    for col in ['Accuracy', 'F1-Score']:
        results_df[col] = results_df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)

    # Sort the results DataFrame by Accuracy in descending order
    results_df.sort_values(by='Accuracy', ascending=False, inplace=True)

    # Display the model comparison header
    st.subheader("Model Comparison")

    # Display the model comparison results in a table WITH horizontal scrolling
    st.dataframe(results_df, use_container_width=True)

    # Reset index to use model names as x-axis labels for plotting
    sorted_accuracy_df = results_df[['Accuracy']].reset_index().rename(columns={'index': 'Model'})

    # Convert 'Accuracy' back to numeric for plotting
    sorted_accuracy_df['Accuracy'] = pd.to_numeric(sorted_accuracy_df['Accuracy'])

    # Create Seaborn figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust figure size as needed

    # Create Seaborn bar plot with custom palette
    palette = sns.color_palette("hls", len(sorted_accuracy_df))
    sns.barplot(data=sorted_accuracy_df, x='Model', y='Accuracy', palette=palette, ax=ax)

    # Customize the plot (optional)
    ax.set_title('Model Accuracy Comparison')  # Set title on the axes
    ax.set_ylabel('Accuracy')
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability

    # Display the Seaborn plot in Streamlit
    st.pyplot(fig)

# Function 5: a function to display and compare feature results in a Streamlit app
def display_and_compare_features(feature_model_results):
    """
    Displays and compares feature results in a Streamlit app.

    Args:
        feature_model_results (dict): A dictionary containing feature results, where keys are feature names and values are dictionaries of metrics.
    """

    # Create a DataFrame from the results dictionary
    features_df = pd.DataFrame(feature_model_results).transpose()

    # Define the new order of columns
    new_column_order = ['Features', 'Target', 'Accuracy', 'F1-Score', 'Model Type', 'k-Value', 'Evaluation Method']
    features_df = features_df[new_column_order]

    # Set Pandas display options to prevent truncation
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # Format Accuracy and F1-Score
    for col in ['Accuracy', 'F1-Score']:
        features_df[col] = features_df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)

    # Sort by Accuracy
    features_df.sort_values(by='Accuracy', ascending=False, inplace=True)

    # Display the results in a table
    st.subheader("Feature Comparison")
    st.dataframe(features_df, use_container_width=True)

    # Reset index to use feature names as x-axis labels for plotting
    sorted_accuracy_features = features_df[['Accuracy']].reset_index().rename(columns={'index': 'Feature'})

    # Convert 'Accuracy' back to numeric for plotting
    sorted_accuracy_features['Accuracy'] = pd.to_numeric(sorted_accuracy_features['Accuracy'])

    # Create Seaborn figure and axes
    fig, ax = plt.subplots(figsize=(10, 5)) 
    # Create Seaborn bar plot with custom palette
    palette = sns.color_palette("hls", len(sorted_accuracy_features))
    sns.barplot(data=sorted_accuracy_features, x='Feature', y='Accuracy', hue='Feature', dodge=False, palette=palette, ax=ax, legend=False) 
    # Customize the plot
    ax.set_title('Feature Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    # Display the Seaborn plot in Streamlit
    st.pyplot(fig)
