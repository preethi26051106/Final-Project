import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# df =pd.read_csv("diamonds.csv")
# print(df.head())
# stand scalar thru code we have to list these numeric columns using for loop


# Load the Diamonds dataset
df = sns.load_dataset('diamonds')

# Display the first few rows to understand the structure of the dataset
print(df.head())

"""creating a list of the numerical data seperately"""

# only_numerical_data = []
# for col in df:
#     if df[col].dtype == "O":
#         only_numerical_data.append(col)
# print(only_numerical_data)

# Get a list of numeric columns dynamically
numeric_columns = df.select_dtypes(include=['number']).columns

# Display the numeric columns
print("\nNumeric columns in the Diamonds dataset:")
print(numeric_columns)
#
# numeric_columns = ['carat','depth','table','price','x','y','z']
numeric_data =df[numeric_columns]

#Standardizing the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

#Applyiong PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

print(pca_result)

pca_columns = ["PC1", "PC2","PC3","PC4","PC5","PC6","PC7"][:pca_result.shape[1]]
print(pca_columns)
pca_df=pd.DataFrame(pca_result,columns=pca_columns)
print(pca_df)

