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
        logger.info("Analyzing univariate distributions...")
        
        # Get numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Create histograms for numerical columns
        for col in numerical_cols:
            fig = px.histogram(self.data, x=col, 
                             title=f'Distribution of {col}',
                             marginal='box',  # Add box plot on the margin
                             nbins=50)
            fig.show()
            
            # Add QQ plot for normality check
            fig = go.Figure()
            qq = stats.probplot(self.data[col].dropna(), dist="norm")
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers'))
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], mode='lines'))
            fig.update_layout(title=f'Q-Q Plot for {col}')
            fig.show()
        
        # Create bar charts for categorical columns
        for col in categorical_cols:
            value_counts = self.data[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Distribution of {col}',
                        labels={'x': col, 'y': 'Count'})
            fig.show()
            
            # Add percentage distribution
            percentage_dist = (value_counts / len(self.data) * 100).round(2)
            fig = px.pie(values=percentage_dist.values, 
                        names=percentage_dist.index,
                        title=f'Percentage Distribution of {col}')
            fig.show()

            # Analyze relationships between TotalPremium and TotalClaims by ZipCode
            if 'TotalPremium' in self.data.columns and 'TotalClaims' in self.data.columns and 'ZipCode' in self.data.columns:
                # Create scatter plot with ZipCode as color
                fig = px.scatter(self.data, 
                               x='TotalPremium', 
                               y='TotalClaims',
                               color='ZipCode',
                               title='Relationship between TotalPremium and TotalClaims by ZipCode',
                               labels={'TotalPremium': 'Total Premium', 'TotalClaims': 'Total Claims'},
                               trendline='ols')  # Add trend line
                fig.show()

                # Calculate correlation matrix for numerical columns
                numerical_cols = ['TotalPremium', 'TotalClaims']
                corr_matrix = self.data[numerical_cols].corr()
                
                # Create correlation heatmap
                fig = px.imshow(corr_matrix,
                              labels=dict(color='Correlation'),
                              title='Correlation Matrix: TotalPremium vs TotalClaims',
                              color_continuous_scale='RdBu')
                fig.show()

                # Group by ZipCode and calculate mean values
                zipcode_analysis = self.data.groupby('ZipCode')[['TotalPremium', 'TotalClaims']].mean()
                
                # Create bar chart comparing average values by ZipCode
                fig = px.bar(zipcode_analysis,
                           title='Average TotalPremium and TotalClaims by ZipCode',
                           barmode='group',
                           labels={'value': 'Average Amount', 'variable': 'Metric'})
                fig.show()
        
    
            
    