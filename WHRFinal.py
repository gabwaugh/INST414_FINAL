import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the WHR dataset
df = pd.read_csv('WHRData.csv')

# Data Cleaning
# Remove rows with missing values
df = df.dropna()

# Select features for clustering and regression analysis
features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
target = 'Life Ladder'

X = df[features]
y = df[target]

# Standardize the data for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering and Outlier Detection
# Use the Elbow Method to determine optimal number of clusters
sse = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method to Determine Optimal k')
plt.show()

# Perform KMeans Clustering with optimal k (e.g., k=4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate distances to cluster centroids to identify outliers
df['Distance_to_Centroid'] = np.min(kmeans.transform(X_scaled), axis=1)
outlier_threshold = df['Distance_to_Centroid'].quantile(0.95)
df['Outlier'] = df['Distance_to_Centroid'] > outlier_threshold

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Log GDP per capita', y='Social support', hue='Cluster', palette='viridis', s=100)
plt.title('Clusters of Countries (World Happiness Report)')
plt.xlabel('Log GDP per capita')
plt.ylabel('Social Support')
plt.legend(title='Cluster')
plt.show()

# Regression Analysis
# Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and fit Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

# Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Linear Regression Performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R-squared: {r2:.3f}")

# Visualize true vs predicted values for Linear Regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("True Life Ladder Score")
plt.ylabel("Predicted Life Ladder Score")
plt.title("True vs Predicted Life Ladder Score (Linear Regression)")
plt.show()

# Visualize the residuals for Linear Regression using Matplotlib
plt.figure(figsize=(8, 5))

# Calculate residuals
residuals = y_test - y_pred

# Create a scatter plot for residuals
plt.scatter(y_test, residuals, alpha=0.6, color='g')

# Add a horizontal line at 0 to indicate the baseline
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

# Label the axes and title
plt.xlabel("True Life Ladder Score")
plt.ylabel("Residuals")
plt.title("Residuals of Linear Regression")

# Show the plot
plt.show()
