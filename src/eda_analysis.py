#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import logging as logger
from scipy import stats

class InsuranceEDA:
    def __init__(self):
        """
        Initialize the EDA analysis class.
        """
        
        # Initialize dataframe
        self.data = None
    
    def load_data(self, data_path) -> None:
        """Load and perform initial data inspection."""
        logger.info("Loading data...")
        # self.data = pd.read_csv(data_path, sep='|', encoding='utf-8', low_memory=False)
        self.data = pd.read_parquet(data_path)
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'])
        return self.data
    
    
    def check_missing_values(self):
        """
        Analyze missing values in the dataset.
        
        Returns:
            pd.DataFrame: Summary of missing values
        """
        try:
            # Calculate missing values count and percentage
            missing_values = pd.DataFrame({
                'Missing Count': self.data.isnull().sum(),
                'Missing Percentage': (self.data.isnull().sum() / len(self.data)) * 100
            })
            
            # Sort by missing percentage in descending order
            missing_values = missing_values.sort_values('Missing Percentage', ascending=False)
            
            # Filter to show only columns with missing values
            missing_values = missing_values[missing_values['Missing Count'] > 0]
            
            print("\n--------------------------------Missing Values Analysis--------------------------------")
            print(f"Total number of columns with missing values: {len(missing_values)}")
            print("\nMissing values summary:")
            
            return missing_values
            
        except Exception as e:
            print(f"Error in check_missing_values: {str(e)}")
            raise
    
    
    def summarize_data(self):
        """
        Generate comprehensive summary statistics and data structure analysis.
        """
        try:
            # Display basic information about the dataset
            print("\n--------------------------------Dataset Information--------------------------------")
            print(f"Number of rows: {self.data.shape[0]}")
            print(f"Number of columns: {self.data.shape[1]}")
            
            # Display data types and non-null counts
            print("\n--------------------------------Data Types and Non-Null Counts--------------------------------")
            print(self.data.info())
            
            # Calculate descriptive statistics for numerical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            print("\n--------------------------------Descriptive Statistics for Numerical Features--------------------------------")
            desc_stats = self.data[numerical_cols].describe()
            print(desc_stats)
            
            # Calculate additional variability metrics
            print("\n--------------------------------Additional Variability Metrics--------------------------------")
            variability_metrics = pd.DataFrame({
                'Variance': self.data[numerical_cols].var(),
                'Skewness': self.data[numerical_cols].skew(),
                'Kurtosis': self.data[numerical_cols].kurtosis(),
                'IQR': self.data[numerical_cols].quantile(0.75) - self.data[numerical_cols].quantile(0.25)
            })
            
            # Analyze categorical columns
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            print("\n--------------------------------Categorical Columns Analysis--------------------------------")
            for col in categorical_cols:
                print(f"\nUnique values in {col}: {self.data[col].nunique()}")
                print(f"Value counts for {col}:")
                print(self.data[col].value_counts().head())
            
            # Check for date columns and their format
            date_cols = self.data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                print("\n--------------------------------Date Columns Analysis--------------------------------")
                for col in date_cols:
                    print(f"\nDate range for {col}:")
                    print(f"Start: {self.data[col].min()}")
                    print(f"End: {self.data[col].max()}")
            
            return desc_stats, variability_metrics
            
        except Exception as e:
            logger.error(f"Error in data summarization: {str(e)}")
            raise

    
    def analyze_univariate_distributions(self) -> None:
        """Analyze and visualize univariate distributions of variables."""
        print("Analyzing univariate distributions...")
        
        # Get numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Create histograms for numerical columns
        for col in numerical_cols:
            plt.figure(figsize=(8, 6))
            sns.hitstplot(data = self.data, x = col)
            plt.title(f"Distribution of {col}")
            plt.xlabel(f" Values{col}")
            plt.ylabel("Count")
            plt.legend(title=f"{col}")
            plt.show()
        
        # Create bar charts for categorical columns
        for col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=self.data, x=col)
            plt.title(f"Count for {col}")
            plt.xlabel(f"{col}")
            plt.ylabel("Counts")
            plt.show()
    
    def analyze_premium_claims_relationships(self) -> None:
        """Analyze relationships between TotalPremium and TotalClaims by PostalCode."""
        print("Analyzing relationships between TotalPremium and TotalClaims...")
        
        # Create scatter plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.data, x='TotalPremium', y='TotalClaims', hue='PostalCode', alpha=0.6)
        plt.title('Relationship between TotalPremium and TotalClaims by PostalCode')
        plt.xlabel('TotalPremium')
        plt.ylabel('TotalClaims')
        plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Calculate correlation matrix for numerical columns
        numerical_cols = ['TotalPremium', 'TotalClaims']
        corr_matrix = self.data[numerical_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix: TotalPremium vs TotalClaims')
        plt.tight_layout()
        plt.show()

        # Group by PostalCode and calculate mean values
        postal_code_analysis = self.data.groupby('PostalCode')[['TotalPremium', 'TotalClaims']].mean()
        print("\nAverage TotalPremium and TotalClaims by PostalCode:")
        return postal_code_analysis
    
    def analyze_geographic_trends(self) -> None:
        """Analyze trends in insurance data across different geographic regions."""
        print("Analyzing geographic trends in insurance data...")
        
        # Group by PostalCode and analyze key metrics
        geo_analysis = self.data.groupby('PostalCode').agg({
            'TotalPremium': 'mean',
            'TotalClaims': 'mean',
            'CoverType': lambda x: x.value_counts().index[0],  # Most common cover type
            'AutoMake': lambda x: x.value_counts().index[0],   # Most common auto make
            'AutoYear': 'mean'
        }).round(2)
        
        # Plot average premium by postal code
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.data, x='PostalCode', y='TotalPremium')
        plt.title('Average Premium by Postal Code')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot distribution of cover types by postal code
        plt.figure(figsize=(12, 6))
        cover_type_by_postal = pd.crosstab(self.data['PostalCode'], self.data['CoverType'])
        cover_type_by_postal.plot(kind='bar', stacked=True)
        plt.title('Cover Type Distribution by Postal Code')
        plt.xlabel('Postal Code')
        plt.ylabel('Count')
        plt.legend(title='Cover Type')
        plt.tight_layout()
        plt.show()
        
        # Plot average auto year by postal code
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x='PostalCode', y='AutoYear')
        plt.title('Auto Year Distribution by Postal Code')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return geo_analysis
    
    def detect_outliers(self) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in numerical columns using box plot analysis.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing outlier information for each numerical column
        """
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            # Calculate Q1, Q3, and IQR
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = self.data[
                (self.data[col] < lower_bound) | 
                (self.data[col] > upper_bound)
            ]
            
            # Store outlier information
            outlier_info[col] = outliers
            
            # Create box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col])
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            plt.show()
            
            # Print summary statistics
            print(f"\nOutlier Analysis for {col}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Lower bound: {lower_bound:.2f}")
            print(f"Upper bound: {upper_bound:.2f}")
            print(f"Percentage of outliers: {(len(outliers) / len(self.data)) * 100:.2f}%")
        
        return outlier_info

                    
    def create_insightful_visualizations(self) -> Dict[str, plt.Figure]:
            
        """
        Create three insightful visualizations based on EDA findings.
        
        Returns:
            Dict[str, plt.Figure]: Dictionary containing the created plots
        """
               
        # 1. Time Series Analysis with Trend and Seasonality
        plt.figure(figsize=(15, 6))
        monthly_avg = self.data.groupby('TransactionMonth')['PremiumAmount'].mean()
        sns.lineplot(data=monthly_avg, marker='o')
        plt.title('Monthly Premium Amount Trends', fontsize=14, pad=20)
        plt.xlabel('Transaction Month', fontsize=12)
        plt.ylabel('Average Premium Amount', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 2. Correlation Heatmap with Custom Styling
        plt.figure(figsize=(12, 8))
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
        plt.show()
        
        # 3. Distribution of Premium Amount by Category
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.data, x='PolicyType', y='PremiumAmount', 
                      palette='Set3', inner='quartile')
        plt.title('Premium Amount Distribution by Policy Type', fontsize=14, pad=20)
        plt.xlabel('Policy Type', fontsize=12)
        plt.ylabel('Premium Amount', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.show()
        

            
    