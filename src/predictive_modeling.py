import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceRiskModel:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42)
        }
        self.preprocessor = None
        self.best_model = None
        self.feature_names = None
        
    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        """
        Prepare data for modeling by handling missing values and encoding categorical variables
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Store feature names for later use
        self.feature_names = list(numeric_features) + list(categorical_features)
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train all models and store their performance
        """
        results = {}
        
        for name, model in self.models.items():
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            results[name] = pipeline
            
        return results
    
    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate all models using RMSE and R-squared
        """
        results = {}
        
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'R-squared': r2
            }
            
        return results
    
    def analyze_feature_importance(self, model, X_test):
        """
        Analyze feature importance using SHAP values
        """
        # Get feature names after preprocessing
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Get SHAP values
        explainer = shap.TreeExplainer(model.named_steps['model'])
        shap_values = explainer.shap_values(model.named_steps['preprocessor'].transform(X_test))
        
        # Plot SHAP summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title('Feature Importance (SHAP Values)')
        plt.tight_layout()
        
        return shap_values, feature_names

def main():
    # Load your data here
    # df = pd.read_csv('path_to_your_data.csv')
    
    # Initialize the model
    model = InsuranceRiskModel()
    
    # Prepare data
    # X_train, X_test, y_train, y_test = model.prepare_data(df, target_col='TotalClaims')
    
    # Train models
    # trained_models = model.train_models(X_train, y_train)
    
    # Evaluate models
    # results = model.evaluate_models(trained_models, X_test, y_test)
    
    # Analyze feature importance for the best model
    # shap_values, feature_names = model.analyze_feature_importance(trained_models['best_model'], X_test)

if __name__ == "__main__":
    main() 