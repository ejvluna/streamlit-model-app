import streamlit as st
import pandas as pd
from model_functions import train_and_evaluate_gnb, train_and_evaluate_knn

# Load your dataset
df = pd.read_excel(r"C:\Users\Admin\OneDrive\Massey University\Semester 3\Data Wrangling & Machine Learning\Assignments\A3\Sample.xlsx")

st.title('Interactive Model Training')

# Column Selection
st.write("Select columns to include in model")
columns = df.columns.tolist()
selected_columns = st.multiselect("Columns", columns)
df_subset = df[selected_columns]

if not df_subset.empty:

    # Feature and Target Selection
    st.sidebar.header('Feature and Target Selection')
    selected_features = st.sidebar.multiselect("Features", selected_columns)
    target_variable = st.sidebar.selectbox("Target", selected_columns)

    if selected_features and target_variable:
        # Check for identical feature and target
        if target_variable in selected_features:
            st.error("Target variable cannot be the same as a feature. Please select a different target.")

        else:
            # Model Selection
            st.sidebar.header('Model Selection')
            classifier_name = st.sidebar.selectbox('Choose Classifier', ('Gaussian Naive Bayes', 'kNN'))

            # Evaluation Method Selection
            evaluation_method = st.sidebar.selectbox('Choose Evaluation Method', ('Train-Test Split', '10-Fold Cross-Validation'))

            if classifier_name == 'kNN':
                # Option to choose or optimize k
                k_selection_method = st.sidebar.radio("k-Value Selection", ("Choose k", "Find Optimal k"))
                if k_selection_method == "Choose k":
                    k = st.sidebar.slider('Select k for kNN', 3, 21, 5, 2)  
                else:
                    k = None

            # "Run Model" Button
            if st.button("Run Model"):
                if classifier_name == 'Gaussian Naive Bayes':
                    train_and_evaluate_gnb(df_subset, selected_features, target_variable, evaluation_method)
                elif classifier_name == 'kNN':
                    train_and_evaluate_knn(df_subset, selected_features, target_variable, evaluation_method, k) 

else:
    st.write("Please select at least one column to proceed.")  # More specific message
