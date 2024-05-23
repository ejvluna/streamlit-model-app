import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# (Your interactive_model_training function from the previous response)

# Function to create the interactive Streamlit app
def interactive_model_training(df):
    st.title('Interactive Model Training')

    # Display available features and target variable
    st.sidebar.header('Feature Selection')
    st.sidebar.subheader('Available Features:')
    available_features = df.columns.tolist()  # Get all column names as features
    selected_features = st.sidebar.multiselect('Select Features', available_features)

    st.sidebar.subheader('Target Variable:')
    target_variable = st.sidebar.selectbox('Select Target', available_features)

    if selected_features and target_variable:
        # Create feature matrix X and target variable y
        X = df[selected_features]
        y = df[target_variable]

        # Model selection
        st.sidebar.header('Model Selection')
        classifier_name = st.sidebar.selectbox('Choose Classifier', ('Gaussian Naive Bayes', 'kNN'))

        if classifier_name == 'Gaussian Naive Bayes':
            model = GaussianNB()
        elif classifier_name == 'kNN':
            k = st.sidebar.slider('Select k for kNN', 1, 20, step=2)
            model = KNeighborsClassifier(n_neighbors=k)
        else:
            st.error("Invalid classifier selected.")

        if model:
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Training
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Display results (you can add more detailed evaluation here)
            st.subheader('Model Performance')
            st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')

            # Add additional model-specific metrics (e.g., F1-score) if needed

# Load your dataset
df = pd.read_excel(r"C:\Users\Admin\OneDrive\Massey University\Semester 3\Data Wrangling & Machine Learning\Assignments\A3\Sample.xlsx")

# Display available columns and let the user select features and target variable
st.write("Select columns to include in model")
columns = df.columns.tolist()
selected_columns = st.multiselect("Columns", columns)

# Create the subset DataFrame
df_subset = df[selected_columns]

# Run the interactive model training app
if not df_subset.empty:  # Check if any columns were selected
    interactive_model_training(df_subset)
else:
    st.write("Please select at least one column to proceed.")
