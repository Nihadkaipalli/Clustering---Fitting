#Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats

# Define the curve fit function
def curve_fit_function(x, a, b, c):
    """
    Exponential curve fit function.

    Parameters:
        x (array): Independent variable.
        a, b, c (float): Parameters to be optimized.

    Returns:
        array: Fitted values based on the curve fit function.
    """
    return a * np.exp(b * (x - x.iloc[0])) + c

# Function to estimate confidence range
def err_ranges(x, pcov, *popt, confidence=0.95):
    """
    Estimate confidence range for the curve fit.

    Parameters:
        x (array): Independent variable.
        pcov (array): Covariance matrix from curve_fit.
        *popt (float): Optimized parameters.
        confidence (float): Confidence level (default is 0.95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    perr = np.sqrt(np.diag(pcov))
    t_value = scipy.stats.t.ppf((1 + confidence) / 2, len(x) - len(popt))
    lower_bound = curve_fit_function(x, *(popt - t_value * perr))
    upper_bound = curve_fit_function(x, *(popt + t_value * perr))
    return lower_bound, upper_bound

# Function to read and clean data
def read_and_clean_data(file_path, output_file):
    """
    Read and clean data from a CSV file, and save the cleaned data.

    Parameters:
        file_path (str): Path to the CSV file.
        output_file (str): Path to save the cleaned data.

    Returns:
        DataFrame: Cleaned data.
    """
    df_melt = pd.read_csv(file_path).melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Year', value_name='Value'
    )

    df_pivot = df_melt.pivot_table(
        index=['Country Name', 'Country Code', 'Year'],
        columns='Indicator Name', values='Value'
    ).reset_index()

    cleaned_data = df_pivot.fillna(df_pivot.mean())
    cleaned_data.to_csv(output_file)
    return cleaned_data

# Function to perform clustering and calculate silhouette score
def perform_clustering(data, num_clusters):
    """
    Perform KMeans clustering on numeric data and calculate silhouette score.

    Parameters:
        data (DataFrame): Input data.
        num_clusters (int): Number of clusters for KMeans.

    Returns:
        tuple: Data with cluster labels, KMeans model.
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    numeric_data = data[numeric_columns]

    imputer = SimpleImputer(strategy='mean')  
    numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_columns)

    normalized_data = (numeric_data_imputed - numeric_data_imputed.mean()) / numeric_data_imputed.std()

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(normalized_data)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(normalized_data, kmeans.labels_)
    print(f"Silhouette Score for {num_clusters} clusters: {silhouette_avg}")

    return data, kmeans

# Function to plot clusters
def plot_clusters(data, kmeans, x_column, y_column, title):
    """
    Plot clusters and cluster centers.

    Parameters:
        data (DataFrame): Input data with cluster labels.
        kmeans (KMeans): Fitted KMeans model.
        x_column (str): Column for the x-axis.
        y_column (str): Column for the y-axis.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 8))
    for cluster in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data[x_column], cluster_data[y_column], label=f'Cluster {cluster}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Cluster Centers')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    plt.legend()
    plt.show()

# Function for curve fitting and prediction
def curve_fit_and_predict(data, country_name, x_column, y_column):
    """
    Perform curve fitting and prediction for a specific country.

    Parameters:
        data (DataFrame): Input data.
        country_name (str): Name of the country.
        x_column (str): Column for the x-axis.
        y_column (str): Column for the y-axis.
    """
    country_data = data[data['Country Name'] == country_name]
    x_data = country_data['Year'].astype(int)
    y_data = country_data[y_column].astype(float)

    # Curve fitting and prediction
    popt, pcov = curve_fit(curve_fit_function, x_data, y_data, maxfev=15000)
    
    prediction_years = pd.Series(range(1990, 2031))
    predicted_values = curve_fit_function(prediction_years, *popt)
    
    # Estimate confidence range
    lower_bound, upper_bound = err_ranges(prediction_years, pcov, *popt)

    # Plotting the Actual and Predicted Data
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o-', label=f'{country_name} Actual Data')
    plt.plot(prediction_years, predicted_values, 'x-', label=f'{country_name} Predicted (1990-2030)')
    plt.xlabel('Year')
    plt.ylabel(f'{y_column}')
    plt.title(f'{y_column} for {country_name} with Extended Curve Fit (1990-2030)')
    plt.legend()
    plt.grid(True)
    plt.show()
       

    # Plotting the best-fitting function with confidence range
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o-', label=f'{country_name} Actual Data')
    plt.plot(prediction_years, curve_fit_function(prediction_years, *popt), 'x-', label=f'{country_name} Best Fit')
    plt.fill_between(prediction_years, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Range')
    plt.xlabel('Year')
    plt.ylabel(f'{y_column}')
    plt.title(f'{y_column} for {country_name} with Confidence Range (1990-2030)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Data processing and clustering for CO2 emissions
co2_data = read_and_clean_data('CO2 Emission.csv', 'co2_cleaned.csv')
forest_data = read_and_clean_data('Forest Area.csv', 'forest_cleaned.csv')

# Merging the datasets
merged_data = pd.merge(co2_data, forest_data, on=['Country Name', 'Country Code', 'Year'])

# Clustering and Silhouette Score Calculation
merged_data, kmeans_co2 = perform_clustering(merged_data, num_clusters=5)

# Plotting Clusters for CO2 emissions
plot_clusters(merged_data, kmeans_co2, 'Forest area (% of land area)', 'CO2 emissions (kt)',
              'Clustering of Countries based on CO2 Emission and Forest area (% of land area)')

# Curve fitting and prediction for Bolivia and Central African Republic
curve_fit_and_predict(merged_data, 'Bolivia', 'Year', 'CO2 emissions (kt)')
curve_fit_and_predict(merged_data, 'Central African Republic', 'Year', 'CO2 emissions (kt)')

# Curve fitting and prediction for Forest Area for Bolivia and Central African Republic
curve_fit_and_predict(merged_data, 'Bolivia', 'Year', 'Forest area (% of land area)')
curve_fit_and_predict(merged_data, 'Central African Republic', 'Year', 'Forest area (% of land area)')
