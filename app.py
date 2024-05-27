# Import required libraries
import streamlit as st # to create web app
import pandas as pd # to work with data
import io  # Additional import for working with file-like objects
import numpy as np # for numerical operations
from model_functions import train_and_evaluate_gnb, train_and_evaluate_knn # to train and evaluate models
from model_functions import display_and_compare_models, display_and_compare_features # to compare models and features
import openpyxl # to read excel files in github
import requests # to read excel files in github

# Load the dataset from the URL
file_url = "https://raw.githubusercontent.com/ejvluna/streamlit-model-app/main/TB_Burden_Country_Cleaned.xlsx"
r = requests.get(file_url)
df = pd.read_excel(io.BytesIO(r.content))  # Read Excel data from BytesIO object


# === Phase 1: Prep ===
# Load the dataset
df = pd.read_excel(r"C:\Users\Admin\OneDrive\Massey University\Semester 3\Data Wrangling & Machine Learning\Assignments\A3\TB_Burden_Country_Cleaned.xlsx")
# Check if 'model_results' and 'model_counter' are not in session state (i.e. first run of the app)
if 'model_results' not in st.session_state:
    # If so, then create an empty dictionary to store results for each model created
    st.session_state.model_results = {}
# Check if 'model_counter' is not in session state (i.e. first run of the app)
if 'model_counter' not in st.session_state:
    # If so, then create a counter to track the number of models created
    st.session_state.model_counter = 1
# Load the model results and model counter from session state to ensure persistence across model runs
model_results = st.session_state.model_results
model_counter = st.session_state.model_counter
# Check if 'feature_model_results' and 'feature_model_counter' are not in session state (i.e. first run of the app)
if 'feature_model_results' not in st.session_state:
    # If so, then create an empty dictionary to store results for each feature evaluated
    st.session_state.feature_model_results = {}
# Check if 'feature_model_counter' is not in session state (i.e. first run of the app)
if 'feature_model_counter' not in st.session_state:
    # If so, then create a counter to track the number of features evaluated
    st.session_state.feature_model_counter = 1
# Load the model results and model counter from session state to ensure persistence across model runs
feature_model_results = st.session_state.feature_model_results
feature_model_counter = st.session_state.feature_model_counter

# === Phase 2: User Interface ===

# Section 1: Feature & Model Configuration

# Display the app title
st.title('Interactive Model Training')
# Display the interactive widgets for feature and model selection in the sidebar
st.sidebar.header('Feature and Target Selection')
# Target selection: Hardcoded to 'Region'
target_variable = 'Region'
st.sidebar.write(f"**Target Variable:** {target_variable}") 
# Feature Selection (Dropdown) - No Default Selection
feature_options = [col for col in df.columns if col != target_variable]  # Exclude target
selected_features = st.sidebar.multiselect("Features", feature_options) # No default selection


# Section 2: Dataset Display
st.header("Dataset Preview")

# Display DataFrame using st.dataframe for interactive features
st.dataframe(df)  # Shows full dataset by default

# Option to show only selected features:
if selected_features:  # Check if any features are selected
    st.subheader("Selected Features Preview")
    st.dataframe(df[selected_features])

# When at least one feature is selected
if selected_features:
    # Create df_subset DataFrame with selected feature/s and target
    df_subset = df[[target_variable] + selected_features]  
    # When the target and at least one feature is selected
    if selected_features and target_variable:
        # Check for identical feature and target
        if target_variable in selected_features:
            # Display error message if target variable is the same as a feature
            st.error("Target variable cannot be the same as a feature. Please select a different target.")
        # Otherwise, proceed with model selection
        else:
            # Display the model selection header
            st.sidebar.header('Model Selection')
            # Display the model options in a selectbox: Gaussian Naive Bayes and kNN and store the selected classifier
            classifier_name = st.sidebar.selectbox('Choose Classifier', ('Gaussian Naive Bayes', 'kNN'))
            # Display the evaluation method options in a selectbox: Train-Test Split and 10-Fold Cross-Validation and store the selected evaluation method
            evaluation_method = st.sidebar.selectbox('Choose Evaluation Method', ('Train-Test Split', '10-Fold Cross-Validation'))
            # kNN k-Value Selection Logic
            if classifier_name == 'kNN':
                # Display a slider to select k and specify the range, default value, and step size
                k_selection_method = st.sidebar.radio("k-Value Selection", ("Choose k", "Find Optimal k"))
                # If "Choose k" is selected, then
                if k_selection_method == "Choose k":
                    # Display a slider to select k and specify the range, default value, and step size
                    k = st.sidebar.slider('Select k for kNN', 3, 21, 5, 2)  
                # Otherwise, if "Find Optimal k" is selected, then
                else:
                    # Set k to None (i.e will need to find optimal k)
                    k = None

            # Sidebar
            st.sidebar.header("Model Creation")  # Optional header in the sidebar

            # Move the button to the sidebar
            if st.sidebar.button("Create Model"):
            # "Run Model" Button Logic
            #if st.button("Create Model with Selected Feature/s"):
                # Option 1: User selects Gaussian Naive Bayes classifier
                if classifier_name == 'Gaussian Naive Bayes':
                    # Call the train_and_evaluate_gnb function to train and evaluate the model
                    model_results, model_counter = train_and_evaluate_gnb(df_subset, selected_features, target_variable, evaluation_method, model_results, model_counter)
                # Option 2: User selects kNN classifier
                elif classifier_name == 'kNN':
                    # Call the train_and_evaluate_knn function to train and evaluate the model
                    model_results, model_counter = train_and_evaluate_knn(df_subset, selected_features, target_variable, evaluation_method, model_results, model_counter, k)
                # Update session state to store the model results and model counter
                st.session_state.model_results = model_results
                st.session_state.model_counter = model_counter

            # Sidebar: Evaluate Features Button
            st.sidebar.header("Individual Feature Evaluation")
            if st.sidebar.button("Evaluate Features"):
            # Evaluate Features Button Logic
            #if st.button("Evaluate Selected Features Individally"):
                # Check if the selected classifier is Gaussian Naive Bayes or kNN
                if classifier_name in ["Gaussian Naive Bayes", "kNN"]:  
                    # Make a list of all numerical features in the DataFrame
                    features_list = df_subset[selected_features].select_dtypes(include=np.number).columns.tolist()  # Filter selected features before converting to list
                    # Option 1: User selects Gaussian Naive Bayes classifier
                    if classifier_name == 'Gaussian Naive Bayes':
                        # Iterate over each feature in the features_list, and for each do the following:
                        for feature in features_list:
                            # Call the train_and_evaluate_gnb function to train and evaluate the model passing a single feature
                            feature_model_results, feature_model_counter = train_and_evaluate_gnb(
                                df_subset, [feature], target_variable, evaluation_method, feature_model_results, feature_model_counter
                            )
                    # Option 2: User selects kNN classifier
                    elif classifier_name == 'kNN':
                        # Iterate over each feature in the features_list, and for each do the following:
                        for feature in features_list:
                            # Call the train_and_evaluate_knn function to train and evaluate the model passing a single feature
                            feature_model_results, feature_model_counter = train_and_evaluate_knn(
                                df_subset, [feature], target_variable, evaluation_method, feature_model_results, feature_model_counter, k
                    )
                    else:
                        # Display error message if no features are selected
                        st.error("Unsupported classifier for feature evaluation.")
            
            # Update session state to store the feature model results and model counter
            st.session_state.feature_model_results = feature_model_results
            st.session_state.feature_model_counter = feature_model_counter

    else:
        # Display error message if no features are selected
        st.error("Unsupported classifier for feature evaluation.")

else:
    # Display error message if no features are selected
    st.write("Select at least one feature from the sidebar menu to proceed.")


# Section 2: Model and Feature Comparison

# Add spacing for visual separation between sections
st.markdown("---")  

# Create a container to hold the title
with st.container():
    # Display the title for the model and feature comparison section
    st.title("Model Comparison")

# When at least two models are created, display the 'Compare Models' button
if len(model_results) > 1:

    st.write("Click to compare the models created so far.")

    # Create a container to hold the 'Compare Models' button and logic
    if st.button("Compare Models") and len(model_results) > 1:
        # Call the display_and_compare_models function to compare the models
        display_and_compare_models(model_results)   

else:
    # Display error message if less than two models are run and "Compare Models" button is clicked
    st.write("Create at least 2 models to proceed with model comparison.")


# Add spacing for visual separation between sections
st.markdown("---")  
# Create a container to hold the title
with st.container():
    # Display the title for the model and feature comparison section
    st.title("Feature Comparison")
    
# When at least two features are evaluated, display the 'Compare Features' button
if len(feature_model_results) > 1:
    st.write("Click to compare the features evaluated so far.")
    # Create a container to hold the 'Compare Features' button and logic
    if st.button("Compare Features") and len(feature_model_results) > 1:
        # Call the display_and_compare_features function to compare the features
        display_and_compare_features(feature_model_results) 

else:
    # Display error message if less than two models are run and "Compare Models" button is clicked
    st.write("Evaluate at least 2 features to proceed with feature comparison.")
