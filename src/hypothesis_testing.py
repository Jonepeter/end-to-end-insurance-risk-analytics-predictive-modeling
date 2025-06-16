import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import logging
import os
import sys


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the insurance data from the specified file path.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        print(f"Attempting to load data from {file_path}")
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded data with {len(df)} rows")
        print(f"Columns in dataset: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk metrics for the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with calculated risk metrics
    """
    try:
        print("Calculating risk metrics...")
        
        # Check if required columns exist
        required_columns = ['TotalClaims', 'TotalPremium']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate claim frequency (assuming TotalClaims > 0 indicates a claim)
        df['claim_frequency'] = df['TotalClaims'] > 0
        print("Calculated claim frequency")
        
        # Calculate claim severity (average claim amount when claim occurs)
        df['claim_severity'] = df.apply(
            lambda x: x['TotalClaims'] / 1 
            if x['TotalClaims'] > 0 else 0, 
            axis=1
        )
        print("Calculated claim severity")
        
        # Calculate margin
        df['margin'] = df['TotalPremium'] - df['TotalClaims']
        print("Calculated margin")
        
        return df
    except Exception as e:
        print(f"Error calculating risk metrics: {str(e)}")
        raise

def test_province_risk_differences(df: pd.DataFrame) -> Tuple[float, bool]:
    """
    Test for risk differences across provinces using ANOVA.
    
    Args:
        df (pd.DataFrame): Input dataframe with risk metrics
        
    Returns:
        Tuple[float, bool]: (p-value, reject_null)
    """
    try:
        print("Testing province risk differences...")
        if 'Province' not in df.columns:
            raise ValueError("'Province' column not found in dataset")
            
        # Group by province and calculate mean claim frequency
        province_claims = df.groupby('Province')['claim_frequency'].mean()
        print(f"Province claim frequencies: {province_claims.to_dict()}")
        
        # Perform ANOVA test
        f_stat, p_value = stats.f_oneway(
            *[group for _, group in df.groupby('Province')['claim_frequency']]
        )
        
        reject_null = p_value < 0.05
        print(f"Province risk test - p-value: {p_value:.4f}, reject null: {reject_null}")
        return p_value, reject_null
    except Exception as e:
        print(f"Error in province risk test: {str(e)}")
        raise

def test_zipcode_risk_differences(df: pd.DataFrame) -> Tuple[float, bool]:
    """
    Test for risk differences between zip codes using ANOVA.
    
    Args:
        df (pd.DataFrame): Input dataframe with risk metrics
        
    Returns:
        Tuple[float, bool]: (p-value, reject_null)
    """
    try:
        print("Testing zipcode risk differences...")
        if 'PostalCode' not in df.columns:
            raise ValueError("'PostalCode' column not found in dataset")
            
        # Group by zip code and calculate mean claim frequency
        zip_claims = df.groupby('PostalCode')['claim_frequency'].mean()
        print(f"Number of unique zip codes: {len(zip_claims)}")
        
        # Perform ANOVA test
        f_stat, p_value = stats.f_oneway(
            *[group for _, group in df.groupby('PostalCode')['claim_frequency']]
        )
        
        reject_null = p_value < 0.05
        print(f"Zipcode risk test - p-value: {p_value:.4f}, reject null: {reject_null}")
        return p_value, reject_null
    except Exception as e:
        print(f"Error in zipcode risk test: {str(e)}")
        raise

def test_zipcode_margin_differences(df: pd.DataFrame) -> Tuple[float, bool]:
    """
    Test for margin differences between zip codes using ANOVA.
    
    Args:
        df (pd.DataFrame): Input dataframe with risk metrics
        
    Returns:
        Tuple[float, bool]: (p-value, reject_null)
    """
    try:
        print("Testing zipcode margin differences...")
        if 'PostalCode' not in df.columns:
            raise ValueError("'PostalCode' column not found in dataset")
            
        # Group by zip code and calculate mean margin
        zip_margins = df.groupby('PostalCode')['margin'].mean()
        print(f"Number of unique zip codes: {len(zip_margins)}")
        
        # Perform ANOVA test
        f_stat, p_value = stats.f_oneway(
            *[group for _, group in df.groupby('PostalCode')['margin']]
        )
        
        reject_null = p_value < 0.05
        print(f"Zipcode margin test - p-value: {p_value:.4f}, reject null: {reject_null}")
        return p_value, reject_null
    except Exception as e:
        print(f"Error in zipcode margin test: {str(e)}")
        raise

def test_gender_risk_differences(df: pd.DataFrame) -> Tuple[float, bool]:
    """
    Test for risk differences between genders using chi-square test.
    
    Args:
        df (pd.DataFrame): Input dataframe with risk metrics
        
    Returns:
        Tuple[float, bool]: (p-value, reject_null)
    """
    try:
        print("Testing gender risk differences...")
        if 'Gender' not in df.columns:
            raise ValueError("'Gender' column not found in dataset")
            
        # Create contingency table
        contingency = pd.crosstab(df['Gender'], df['claim_frequency'])
        print(f"Gender-claim contingency table:\n{contingency}")
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        reject_null = p_value < 0.05
        print(f"Gender risk test - p-value: {p_value:.4f}, reject null: {reject_null}")
        return p_value, reject_null
    except Exception as e:
        print(f"Error in gender risk test: {str(e)}")
        raise

def visualize_results(df: pd.DataFrame):
    """
    Create visualizations for the hypothesis testing results.
    
    Args:
        df (pd.DataFrame): Input dataframe with risk metrics
    """
    try:
        print(f"Creating visualizations in")
        # Create output directory if it doesn't exist
        
        # Plot 1: Claim Frequency by Province
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Province', y='claim_frequency')
        plt.title('Claim Frequency by Province')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        print("Created province claim frequency plot")
        
        # Plot 2: Claim Severity by Gender
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Gender', y='claim_severity')
        plt.title('Claim Severity by Gender')
        plt.tight_layout()
        plt.show()
        print("Created gender claim severity plot")
        
        # Plot 3: Margin Distribution by Zip Code
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='PostalCode', y='margin')
        plt.title('Margin Distribution by Zip Code')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        print("Created zipcode margin distribution plot")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        raise
