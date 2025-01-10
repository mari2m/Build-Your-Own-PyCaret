#streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) #add the file directory to the python search path
from machine_learning_package import MachineLearningPackage

# --- Initialize Machine Learning Package ---
ml_package = MachineLearningPackage()

# --- Streamlit App Setup ---
st.title("Automated Machine Learning App")
if 'model_trained' not in st.session_state:
  st.session_state.model_trained = False

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

# --- Load Data ---
if uploaded_file is not None:
    if ml_package.load_data(uploaded_file):
        st.success("Data loaded successfully!")
        data = ml_package.data
    else:
        st.error("Failed to load data. Please check file format.")

    # --- Data Display and EDA Options ---
    if data is not None:
        st.subheader("Data Preview:")
        st.dataframe(data.head())

        if st.checkbox("Show Full Data"):
          st.dataframe(data)

        if st.checkbox("Perform Exploratory Data Analysis"):
            with st.spinner('Performing EDA...'):
                ml_package.perform_eda(save_figures=True)
            st.success("EDA Completed!")
            
            if os.path.exists('numerical_histograms.png'):
              st.image('numerical_histograms.png', caption="Numerical Feature Histograms",use_column_width=True)
            if os.path.exists('categorical_value_counts.png'):
              st.image('categorical_value_counts.png', caption="Categorical Feature Value Counts",use_column_width=True)
            if os.path.exists('numerical_pairplot.png'):
              st.image('numerical_pairplot.png', caption="Numerical Features Pair Plot",use_column_width=True)

            

        # --- Target Selection ---
        st.subheader("Select Target Variable:")
        target_variable = st.selectbox("Choose the target variable:", data.columns)

        # --- Model Training ---
        st.subheader("Model Training Options:")
        model_selection = st.radio("Choose model training method:",
                                   ["Auto (AutoML)", "Select Models to Train"])

        model_names_available = [
              'lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge',
                'rf', 'qda', 'ada', 'gbc', 'et', 'xgboost', 'lightgbm', 'catboost'
            ]
        if model_selection == "Select Models to Train":
          selected_models = st.multiselect("Select models to train:",model_names_available)
        else:
          selected_models = 'auto'

        if st.button("Train Models"):
            if target_variable:
              with st.spinner('Training Models...'):
                 try:
                    ml_package.setup_experiment(target_variable=target_variable)
                    ml_package.train_models(models_to_train=selected_models)
                    st.success("Model training completed!")
                    st.session_state.model_trained = True #<--- set the model to trained

                    if model_selection == "Auto (AutoML)":
                       st.write(f"Best Model Identified: {ml_package.best_model}")
                       st.subheader("Model Evaluation:")
                       ml_package.evaluate_model()

                    if model_selection == "Select Models to Train":
                       if ml_package.model_results:
                         st.subheader("Model Evaluation:")
                         for model_name in ml_package.model_results:
                            ml_package.evaluate_model(model_name=model_name)
                 except Exception as e:
                   st.error(f"Error: {e}")
            else:
               st.error("Please specify a target variable")

        # --- Prediction data ---
        st.subheader("Prediction Options:")
        prediction_method = st.radio("Choose prediction data:",
                                   ["New Data Input", "Uploaded Data"])

        if prediction_method == "New Data Input":
          st.write("Provide the input data to make prediction:")
          if data is not None:
            prediction_inputs = {}
            for col in data.columns:
               if col != target_variable:
                  prediction_inputs[col] = st.text_input(f"Enter value for {col}")
            if target_variable: #<-- check for target variable
                if st.button("Make Prediction"):
                    try:
                        new_input_data = {}
                        for col,value in prediction_inputs.items():
                           try:
                             new_input_data[col] = float(value)
                           except:
                             new_input_data[col] = value
                        new_input_data = pd.DataFrame([new_input_data])
                        predictions = ml_package.predict_data(data=new_input_data)
                        if predictions is not None:
                           st.success(f"Predictions : {predictions['prediction_label'].to_list()}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please specify a target variable")
        if prediction_method == "Uploaded Data":
          predict_file = st.file_uploader("Upload your CSV data for prediction", type=["csv"])
          if predict_file is not None:
            predict_data = pd.read_csv(predict_file)
            if st.button("Make Prediction"):
              try:
                predictions = ml_package.predict_data(data=predict_data)
                if predictions is not None:
                   st.success(f"Predictions: {predictions['prediction_label'].to_list()}")
              except Exception as e:
                  st.error(f"Error: {e}")
        # --- Save Model ---
        if st.button("Save Model"):
           if ml_package.best_model: #<--- check if model exists in the class
            try:
              model_filename = "trained_model"
              ml_package.save_model(model_filename)
              st.success(f"Model saved to {model_filename}!")
            except Exception as e:
              st.error(f"Error: {e}")
           else:
              st.error("No model trained, please train first")

        # --- Load Model ---
        if st.button("Load Model"):
           if ml_package.best_model is None: # do not load if a model exists already
              try:
                model_filename = "trained_model"
                ml_package.load_trained_model(model_filename)
                st.success("Model Loaded!")
                st.session_state.model_trained = True #<--- set the model to trained
              except Exception as e:
                st.error(f"Error: {e}")
           else:
              st.error("Model exists already, please save model if needed")