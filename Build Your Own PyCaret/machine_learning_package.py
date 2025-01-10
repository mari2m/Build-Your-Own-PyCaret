import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MachineLearningPackage:
    """
    A general machine learning package for data handling, EDA, and model training
    using PyCaret.
    """

    def __init__(self):
        self.data = None
        self.setup_object = None
        self.best_model = None
        self.model_results = None
        self.model_type = None
        self.target = None
        self.session_id=123
        self.use_gpu=False

    def load_data(self, file_path):
        """Loads data from a CSV file."""
        try:
            self.data = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def perform_eda(self, save_figures=False):
        """Performs Exploratory Data Analysis."""
        if self.data is None:
            print("Error: No data loaded. Please load data first.")
            return

        print("EDA initiated...")

        # Basic Info
        print("\n--- Data Info ---")
        self.data.info()
        print("\n--- Data Description ---")
        print(self.data.describe())

        # Check for missing values
        print("\n--- Missing Values ---")
        print(self.data.isnull().sum())

        # Plot Histograms for numerical features
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        if len(numerical_cols) > 0:
            print("\n--- Numerical Feature Histograms ---")
            num_plots = len(numerical_cols)
            num_cols = 2 if num_plots >= 2 else 1
            num_rows = (num_plots + num_cols - 1) // num_cols
            plt.figure(figsize=(15, 4 * num_rows))
            for i, col in enumerate(numerical_cols):
                plt.subplot(num_rows, num_cols, i + 1)
                sns.histplot(self.data[col], kde=True)
                plt.title(f'Histogram of {col}')
            if save_figures:
                plt.savefig('numerical_histograms.png')
            plt.show()

        # Plot value counts for categorical features
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print("\n--- Categorical Feature Value Counts ---")
            num_plots = len(categorical_cols)
            num_cols = 2 if num_plots >= 2 else 1
            num_rows = (num_plots + num_cols - 1) // num_cols
            plt.figure(figsize=(15, 4 * num_rows))
            for i, col in enumerate(categorical_cols):
                plt.subplot(num_rows, num_cols, i + 1)
                counts = self.data[col].value_counts()
                counts.plot(kind='bar')
                plt.title(f'Value Counts of {col}')
            if save_figures:
                plt.savefig('categorical_value_counts.png')
            plt.show()

        if len(numerical_cols) > 1:
            print("\n--- Pair Plots for Numerical Features ---")
            pair_plot = sns.pairplot(self.data[numerical_cols.to_list()], diag_kind="kde")
            if save_figures:
                pair_plot.savefig('numerical_pairplot.png')
            plt.show()
        print("EDA complete.\n")


    def _determine_model_type(self, target_variable):
        """Determines if the problem is classification or regression."""

        if self.data is None:
           raise Exception("Error: No data loaded. Please load data first.")
        
        if target_variable not in self.data.columns:
           raise Exception(f"Error: Target Variable '{target_variable}' not found in the data.")

        target_series = self.data[target_variable]
        if pd.api.types.is_numeric_dtype(target_series):
          if len(target_series.unique()) <= 10 or len(target_series.unique())/len(target_series) <= 0.05:
             return 'classification'
          return 'regression'
        elif pd.api.types.is_string_dtype(target_series) or pd.api.types.is_categorical_dtype(target_series):
          return 'classification'
        else:
          raise Exception("Target variable not numeric or string type")

    def setup_experiment(self, target_variable,session_id=123,use_gpu=False):
      """Sets up the PyCaret experiment."""

      self.target = target_variable
      self.model_type = self._determine_model_type(target_variable)
      self.session_id=session_id
      self.use_gpu = use_gpu

      print(f"Setting up a {self.model_type} experiment for target variable: {target_variable}...")
      if self.model_type == 'classification':
          self.setup_object = setup(data=self.data, target=target_variable, session_id=self.session_id, use_gpu = self.use_gpu)
      elif self.model_type == 'regression':
          self.setup_object = setup(data=self.data, target=target_variable, session_id=self.session_id, use_gpu = self.use_gpu)
      else:
          raise ValueError("Invalid model_type.")

    def train_models(self, models_to_train='auto'):
        """Trains the specified models or AutoML."""
        if self.setup_object is None:
            raise Exception("Error: Please setup an experiment first.")

        print(f"Training models for {self.model_type}...")

        if models_to_train == 'auto':
            self.best_model = compare_models()
            print(f"Best model identified: {self.best_model}")
        else:
            if isinstance(models_to_train, list):
                self.model_results = {}
                for model_name in models_to_train:
                    if self.model_type == 'classification':
                        try:
                            model = create_model(model_name)
                            self.model_results[model_name] = model
                            print(f"Model '{model_name}' trained.")
                        except Exception as e:
                            print(f"Error training model '{model_name}': {e}")
                    elif self.model_type == 'regression':
                        try:
                            model = create_model(model_name)
                            self.model_results[model_name] = model
                            print(f"Model '{model_name}' trained.")
                        except Exception as e:
                            print(f"Error training model '{model_name}': {e}")
            else:
                print("Invalid input for training models, please specify a model name or 'auto' or list of model names")

        # Check if best_model is set
        if self.best_model is not None:
            print("Model training complete.\n")
        else:
            print("No best model identified. Please check the training process.\n")

    def evaluate_model(self, model_name=None):
       """Evaluates the trained model"""
       if model_name is None:
         if self.best_model is None:
            print("Error: no model selected to evaluate")
         else:
            print(f"Evaluating the best model {self.best_model}")
            evaluate_model(self.best_model)
       elif model_name in self.model_results:
          print(f"Evaluating model {model_name}")
          evaluate_model(self.model_results[model_name])
       else:
          print("Error: model does not exist")

    def predict_data(self, data=None, model_name=None):
        """Predict on the data"""
        if self.best_model is None:
            print("Error: Please train a model first")
            return None
        if data is None:
            print("Error: No data passed for prediction")
            return None
    
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        predictions = predict_model(self.best_model, data)
        return predictions

    def save_model(self, filename):
        """Saves the best trained model"""
        if self.best_model is None:
            print("Error: No model to save. Please train a model first.")
            return
        
        try:
            save_model(self.best_model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

if __name__ == '__main__':
    # Example usage
    ml_package = MachineLearningPackage()

    # Example 1: Classification with Iris dataset
    print("----- Example 1: Classification with Iris dataset -----")
    if ml_package.load_data("iris.csv"):  # Ensure data is loaded successfully
        print("Data loaded successfully.")
        ml_package.perform_eda(save_figures=True)  # Perform EDA, and save generated figures
        
        # Set up experiment and train with auto ML
        try:
            ml_package.setup_experiment(target_variable='species')
            ml_package.train_models()
            ml_package.evaluate_model()

            #Predict on some data
            new_data =  {'sepal_length': [5.1, 6.0], 'sepal_width': [3.5, 3.0], 'petal_length': [1.4, 4.0], 'petal_width': [0.2, 1.0]}
            predictions = ml_package.predict_data(data=new_data)
            print(f"Predictions: {predictions}")
            ml_package.save_model('best_iris_model')
            ml_package.load_trained_model('best_iris_model')

        except Exception as e:
            print(f"Error during classification example: {e}")

        # Clean up the generated image files if they exist
        if os.path.exists('numerical_histograms.png'):
            os.remove('numerical_histograms.png')
        if os.path.exists('categorical_value_counts.png'):
            os.remove('categorical_value_counts.png')
        if os.path.exists('numerical_pairplot.png'):
            os.remove('numerical_pairplot.png')

    else:
        print("Failed to load data for Example 1.")

    # Example 2: Regression with Boston Housing dataset
    print("\n----- Example 2: Regression with Boston Housing dataset -----")
    if ml_package.load_data("boston.csv"):
        print("Data loaded successfully.")
        ml_package.perform_eda(save_figures=True)
         # Set up experiment and train specific models
        try:
             ml_package.setup_experiment(target_variable='medv')
             ml_package.train_models(models_to_train=["lr","rf"]) # train linear regression and random forest models
             #Make prediction
             new_data = {
                 'crim': [0.02731,0.04203], 'zn': [0,28], 'indus': [7.07, 15.04], 'chas': [0,1], 'nox': [0.469, 0.464],
                 'rm': [6.421, 6.761], 'age': [78.9, 42.3], 'dis': [4.9671, 6.336], 'rad': [2, 4], 'tax': [242, 270],
                  'ptratio': [17.8, 18.2], 'b': [396.9, 396.9], 'lstat': [9.14, 5.09]
                 }
             predictions_lr = ml_package.predict_data(data=new_data, model_name="lr")
             print(f"LR Predictions: {predictions_lr}")
             predictions_rf = ml_package.predict_data(data=new_data, model_name="rf")
             print(f"RF Predictions: {predictions_rf}")
             # Evaluate one specific model
             ml_package.evaluate_model(model_name="lr")
             ml_package.save_model('best_boston_model')
             ml_package.load_trained_model('best_boston_model')


        except Exception as e:
            print(f"Error during regression example: {e}")
         # Clean up the generated image files if they exist
        if os.path.exists('numerical_histograms.png'):
            os.remove('numerical_histograms.png')
        if os.path.exists('categorical_value_counts.png'):
            os.remove('categorical_value_counts.png')
        if os.path.exists('numerical_pairplot.png'):
            os.remove('numerical_pairplot.png')

    else:
        print("Failed to load data for Example 2.")

    print("\n----- Examples Complete -----")