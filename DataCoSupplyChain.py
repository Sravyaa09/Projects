#!/usr/bin/env python
# coding: utf-8

# In[6]:


import dmba
from dmba import plotDecisionTree, classificationSummary, regressionSummary 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report


# In[7]:


DataCoSC_df = pd.read_csv(r"C:\Users\sravy\Documents\Fall 2023\Data mining\Project\DataCoSupplyChainDataset.csv", encoding='latin-1')
DataCoSC_df.head()


# ## DATA CLEANING

# In[8]:


DataCoSC_df.isnull().sum()


# In[9]:


DataCoSC_df.drop(['Category Id', 'Customer Email', 'Customer Id', 'Customer City','Customer State', 'Customer Street', 'Customer Zipcode', 'Order City', 'Order Region', 'Order State', 'Customer Password', 'Customer Zipcode', 'Department Id', 
                  'Order Customer Id', 'Order Id', 'Order Item Id', 'Product Card Id', 'Order Zipcode', 'Product Card Id', 
                  'Product Category Id', 'Product Description', 'Product Image', 'Customer Fname', 'Customer Lname'], axis=1, inplace=True)
DataCoSC_df.columns


# In[10]:


DataCoSC_df.columns = [s.strip().replace(' ', '_'). replace('(', '').replace(")", '') for s in DataCoSC_df.columns]
DataCoSC_df.columns


# In[11]:


DataCoSC_df = DataCoSC_df.rename(columns={"order_date_DateOrders": "Order_Date"})
DataCoSC_df = DataCoSC_df.rename(columns={"shipping_date_DateOrders": "Shipping_Date"})


# In[12]:


DataCoSC_df["Order_Date"] = pd.to_datetime(DataCoSC_df["Order_Date"], format='%m/%d/%Y %H:%M')
DataCoSC_df["Order_Date"].unique()


# In[13]:


DataCoSC_df["Shipping_Date"] = pd.to_datetime(DataCoSC_df["Shipping_Date"], format='%m/%d/%Y %H:%M')
DataCoSC_df["Shipping_Date"].unique()


# In[14]:


DataCoSC_df = DataCoSC_df.drop(columns = ["Benefit_per_order", 'Product_Price'])


# In[15]:


DataCoSC_df.dtypes


# In[16]:


DataCoSC_df.shape


# In[17]:


for i in DataCoSC_df.columns:
    if DataCoSC_df[i].dtype=='object':
        print(i,len(DataCoSC_df[i].unique()))


# In[18]:


columns_to_convert = ['Type', 'Delivery_Status', 'Category_Name', 'Customer_Country', 'Customer_Segment', 'Market', 'Order_Country', 'Order_Status', 'Shipping_Mode']
DataCoSC_df[columns_to_convert] = DataCoSC_df[columns_to_convert].astype('category')
DataCoSC_df.dtypes


# In[19]:


data = DataCoSC_df


# ## EXPLORATIVE DATA ANALYSIS

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


plt.rcParams["figure.figsize"] = (15, 8)

def categ_dist(categ_var, y_param, title):
    
    sns.barplot(data= data, x=categ_var, y = y_param, order=data[categ_var].value_counts().index, palette='viridis')
    plt.title(title, size=20)
    plt.show()


# In[22]:


categ_dist('Market', 'Sales', 'Distribution of Sales by Market')
categ_dist('Market', 'Order_Item_Discount_Rate', 'Distribution of Discount Rate by Market')
categ_dist('Market', 'Order_Profit_Per_Order', 'Distribution of Profit by Market')



# In[23]:


categ_dist('Department_Name', 'Sales', 'Distribution of Sales by Department')
categ_dist('Department_Name', 'Order_Item_Discount_Rate', 'Distribution of Discount Rate by Department')
categ_dist('Department_Name', 'Order_Profit_Per_Order', 'Distribution of Profit by Department')


# In[24]:


categ_dist('Customer_Segment', 'Sales', 'Distribution of Sales by Customer Segment')
categ_dist('Customer_Segment', 'Order_Item_Discount_Rate', 'Distribution of Discount Rate by Customer Segment')
categ_dist('Customer_Segment', 'Order_Profit_Per_Order', 'Distribution of Profit by Customer Segment')


# In[25]:


def categ_count(categ_var, title):
    sns.countplot(data = data,x = data[categ_var],order= data[categ_var].value_counts().index, palette='gist_heat')
    plt.title(title, size = 20)
    plt.show()


# In[26]:


categ_count('Shipping_Mode', 'Number of deliveries by each shipping mode')


# In[27]:


data['Profit_Loss'] = data['Order_Profit_Per_Order'].apply(lambda x: 'Profit' if x > 0 else 'Loss')

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Shipping_Mode', hue='Profit_Loss', data=data)
plt.title('Distribution of Profit/Loss by Shipping Mode')
plt.show()


# In[28]:


plt.rcParams["figure.figsize"] = (15, 8)

def categ_shipping(categ_var, title):
    
    sns.countplot(data= data, x=categ_var,hue='Shipping_Mode', order=data[categ_var].value_counts().index, hue_order = data['Shipping_Mode'].value_counts().index, palette='husl')
    plt.title(title, size=20)
    plt.legend(title= "Shipping Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


# In[29]:


categ_shipping("Market",'Distribution of Shipping Modes by Market' )
categ_shipping('Department_Name', 'Distribution of Shipping Modes by Department')
categ_shipping('Customer_Segment', 'Distribution of Shipping Modes by Customer Segment')


# In[30]:


def prod_region(column_to_group, title, n):

    grouped_data = data.groupby(['Product_Name', 'Customer_Country'])[column_to_group].sum().reset_index()
    sorted_grouped_data = grouped_data.sort_values(by=column_to_group, ascending=False)

    top_n = n  
    plt.figure(figsize=(12, 6))
    sns.barplot(x= column_to_group, y='Product_Name', hue='Customer_Country', data=sorted_grouped_data.head(top_n))
    plt.title(title.format(top_n))
    plt.xlabel(column_to_group)
    plt.ylabel('Product Name')
    plt.show()


# In[31]:


prod_region('Sales', 'Top 10 Products Generating Maximum Sales by Region', 10)
prod_region('Order_Item_Discount_Rate', 'Top 10 Products with maximum discount rate by Region', 10)
prod_region('Order_Profit_Per_Order', 'Top 10 Products Generating Maximum Profit by Region', 10)


# In[32]:


import geopandas as gpd


# In[33]:


geometry = gpd.points_from_xy(data['Longitude'], data['Latitude'])
geo_df = gpd.GeoDataFrame(data, geometry=geometry)


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(30,15))
world.plot(ax=ax, color='lightgrey')

geo_df.plot(ax=ax, markersize=10, color='red', alpha=0.5)
plt.title('Geospatial Distribution of Profit')
plt.show()


# ## SCATTERPLOTS FOR NUMERICAL VARIABLES TO DETECT OUTLIERS

# In[34]:


## Sales, Order_Item_Total and Order_Item_Discount_Rate


# In[35]:


from scipy import stats


# In[36]:


def scatter(x_val, y_val):

    sns.regplot(x= x_val, y= y_val, data=data)
    plt.show()


# In[37]:


def z_score_plot(x_val, y_val, threshold, title):
    
    sales_threshold = threshold
   
    data['Z_Score_Sales'] = np.abs(stats.zscore(data['Sales']))

    outliers = data[data['Z_Score_Sales'] > 3]

    plt.scatter(data[x_val], data[y_val], label='Data')
    plt.scatter(outliers[x_val], outliers[y_val], color='red', label='Outliers')

    plt.title(title)
    plt.xlabel(x_val)
    plt.ylabel(y_val)
    plt.axvline(x=sales_threshold, color='orange', linestyle='--', label=f'Sales Threshold: {sales_threshold}')

    plt.legend()
    plt.show()


# In[38]:


scatter('Sales', 'Order_Item_Total')
z_score_plot('Sales', 'Order_Item_Total', 750, 'Scatter Plot of Sales vs. Order Total' )


# In[39]:


scatter('Sales', 'Order_Profit_Per_Order')
z_score_plot('Sales', 'Order_Profit_Per_Order', 750, 'Scatter Plot of Sales vs. Profit' )


# In[40]:


scatter('Order_Item_Discount_Rate','Sales')


# In[41]:


sales_threshold = 750

data['Z_Score_Sales'] = np.abs(stats.zscore(data['Sales']))

outliers = data[data['Z_Score_Sales'] > 3]

plt.scatter(data['Order_Item_Discount_Rate'], data['Sales'], label='Data')
plt.scatter(outliers['Order_Item_Discount_Rate'], outliers['Sales'], color='red', label='Outliers')

plt.title('Scatter Plot of Discount rate vs Sales')
plt.xlabel('Order_Item_Discount_Rate')
plt.ylabel('Sales')
plt.axhline(y=sales_threshold, color='orange', linestyle='--', label=f'Sales Threshold: {sales_threshold}')

plt.legend()
plt.show()


# ## The above plots show that there are significant outliers after the 750 mark. Lets clean the data by deleting the rows with sales above 750

# In[42]:


data_cleaned = data[data['Z_Score_Sales'] <= 3]


# In[43]:


#Shape of data before cleaning outliers
data.shape


# In[44]:


# Shape of data after cleaning outliers
data_cleaned.shape


# In[45]:


## Creating dummy variables for categorical variables
data = pd.get_dummies(data, columns = columns_to_convert, prefix_sep = "_", drop_first = True)
data.columns


# ## CLUSTER ANALYSIS

# In[46]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

columns_for_clustering = ['Order_Item_Discount_Rate', 'Order_Item_Total', 'Order_Profit_Per_Order']

# Extract the relevant subset of the dataframe
data_cleaned_cluster = data_cleaned[columns_for_clustering]


# In[47]:


# Standardize the data
scaler = StandardScaler()
data_cleaned_cluster_scaled = scaler.fit_transform(data_cleaned_cluster)


# In[48]:


# Find the optimal number of clusters using the silhouette score
from sklearn.cluster import MiniBatchKMeans

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_cleaned_cluster_scaled)
    silhouette_scores.append(silhouette_score(data_cleaned_cluster_scaled, cluster_labels))


# In[49]:


# Choose the optimal number of clusters based on the plot
optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Apply k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
data_cleaned['Cluster'] = kmeans.fit_predict(data_cleaned_cluster_scaled)

# Analyze the clusters
cluster_analysis = data_cleaned.groupby('Cluster').agg({
    'Order_Item_Discount_Rate': 'mean',
    'Order_Item_Total': 'sum',
    'Order_Profit_Per_Order': 'sum'
}).reset_index()

# Plotting
plt.figure(figsize=(12, 6))

# Plotting discount rates
plt.subplot(1, 2, 1)
plt.bar(cluster_analysis['Cluster'], cluster_analysis['Order_Item_Discount_Rate'])
plt.title('Average Discount Rate by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Discount Rate')

# Plotting sales and profit
plt.subplot(1, 2, 2)
plt.bar(cluster_analysis['Cluster'], cluster_analysis['Order_Item_Total'], label='Order_Item_Total')
plt.bar(cluster_analysis['Cluster'], cluster_analysis['Order_Profit_Per_Order'], label='Profit')
plt.title('Sales and Profit by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Amount')
plt.legend()

plt.tight_layout()
plt.show()


# In[50]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming you have a DataFrame cluster_analysis with the cluster analysis results

# Extract the cluster labels and relevant columns
cluster_labels = cluster_analysis['Cluster']
sales_column = 'Order_Item_Total'  # Replace with the actual column name
profit_column = 'Order_Profit_Per_Order'  # Replace with the actual column name
discount_rate_column = 'Order_Item_Discount_Rate'  # Replace with the actual column name

# Create StandardScaler instances
#scaler_sales = StandardScaler()
#scaler_profit = StandardScaler()

# Fit and transform the data for Total Sales
#cluster_analysis['Total Sales Original Scale'] = scaler_sales.fit_transform(cluster_analysis[[sales_column]])

Fit and transform the data for Total Profit
cluster_analysis['Total Profit Original Scale'] = scaler_profit.fit_transform(cluster_analysis[[profit_column]])

# Extract the average discount rate for the clusters with max sales and profit
discount_rate_max_sales = cluster_analysis.loc[cluster_analysis['Total Sales Original Scale'].idxmax()][discount_rate_column]
discount_rate_max_profit = cluster_analysis.loc[cluster_analysis['Total Profit Original Scale'].idxmax()][discount_rate_column]

# Print the results
print(f"Cluster with Highest Sales (Cluster {cluster_analysis['Cluster'].iloc[cluster_analysis['Total Sales Original Scale'].idxmax()]}):")
print(f"  Total Sales: ${cluster_analysis['Total Sales Original Scale'].max():,.2f}")
print(f"  Average Discount Rate: {discount_rate_max_sales:.4f}")

print(f"\nCluster with Highest Profit (Cluster {cluster_analysis['Cluster'].iloc[cluster_analysis['Total Profit Original Scale'].idxmax()]}):")
print(f"  Total Profit: ${cluster_analysis['Total Profit Original Scale'].max():,.2f}")
print(f"  Average Discount Rate: {discount_rate_max_profit:.4f}")


# ## DECISION TREES

# In[65]:


from sklearn.tree import DecisionTreeRegressor
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import Image
import os
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'X' is the discount rate and 'y' is the target variable (sales or profit)

# Iterate over clusters
for cluster_id in data_cleaned['Cluster'].unique():
    # Filter data for the current cluster
    cluster_data = data_cleaned[data_cleaned['Cluster'] == cluster_id]
    
    # Extract features and target variable
    X = cluster_data['Order_Item_Discount_Rate'].values.reshape(-1, 1)
    y = cluster_data['Order_Profit_Per_Order'].values  # or 'Order_Profit_Per_Order'
    
    # Fit decision tree model
    model = DecisionTreeRegressor()
    model.fit(X, y)
    
    # Visualize the decision tree (optional)
    dot_data = export_graphviz(model, out_file=None, feature_names=['Discount_Rate'], filled=True, rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)
    
    # Save the decision tree visualization as PNG
    image_path = f'Cluster_{cluster_id}_DecisionTree.png'
    graph.render(image_path, format='png', cleanup=True)
    
    # Display the visualization
    Image(image_path)  # Display the PNG visualization
    
    # Predict using the decision tree (optional)
    predicted_values = model.predict(X)
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = linregress(X.flatten(), y)
    regression_line = intercept + slope * X
    
    # Visualize the relationship (scatter plot with regression line)
    plt.scatter(X, y, label='Actual Values')
    plt.scatter(X, predicted_values, label='Predicted Values', marker='x')
    plt.plot(X, regression_line, color='red', label='Regression Line')
    plt.title(f'Scatter Plot with Decision Tree Prediction: Cluster {cluster_id}')
    plt.xlabel('Discount Rate')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()
    
    # Remove the PDF file if created
    pdf_path = f'Cluster_{cluster_id}_DecisionTree.pdf'
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


# ## Profit and Loss prediction using NEURAL NETWORKS

# In[52]:


# Create the target variable
data_cleaned['Profit_Category'] = np.where(data_cleaned['Order_Profit_Per_Order'] > 0, 'Profit', 'Loss')

# Encode the target variable
label_encoder = LabelEncoder()
data_cleaned['Profit_Category_Encoded'] = label_encoder.fit_transform(data_cleaned['Profit_Category'])

# Features and target
X = data_cleaned.drop(['Profit_Category', 'Profit_Category_Encoded'], axis=1)
y = data_cleaned['Profit_Category_Encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train).toarray()
X_test_preprocessed = preprocessor.transform(X_test).toarray()

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_preprocessed.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_preprocessed, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_preprocessed, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# In[ ]:


# Make predictions on the test set
predictions = model.predict(X_test_preprocessed)
predictions_binary = (predictions > 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(y_test, predictions_binary))


# In[ ]:


# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Update with your actual target variable
target_variable = 'Profit_Category_Encoded'

# Calculate correlation matrix
correlation_matrix = data_cleaned.corr()

# Sort the features based on their correlation with the target variable
correlation_with_target = correlation_matrix[target_variable].sort_values(ascending=False)

# Plot a bar chart to visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index, orient='h', palette='viridis')
plt.title('Feature Importance Based on Correlation with Target')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()


# In[ ]:




