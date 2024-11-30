from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import os
from sklearn import preprocessing
from matplotlib.pyplot import figure

os.environ['OMP_NUM_THREADS'] = '11'


# Open File and create dataframe
df_file = "ip_for_top_acct.csv"

df = pd.read_csv(df_file, \
                 dtype=object)

# Specify which account you want to perform k-means on
df = df[df['account_id'] == 'yvC1QJC9TjyqSUwLMIdpYA']

pd.set_option('display.float_format', lambda x: '%.3f' % x)

ip_df = df['xrealip'].copy()
ip_df = ip_df.reset_index()
del ip_df['index']
ip_df = ip_df.dropna().copy()
ip_df = ip_df.drop_duplicates().copy()

# Open File that contains IP usage for 14 days
df_file1 = 'ip_14days_total.csv'
df1 = pd.read_csv(df_file1, dtype=object)
df1 = df1[df1['account_id'] == 'yvC1QJC9TjyqSUwLMIdpYA']
df2 = df1[['xrealip','ip_total']].copy()
df2 = df2.reset_index()
del df2['index']
df2 = df2.dropna().copy()
df2 = df2.drop_duplicates().copy()



# Exclude IPV6
for i in ip_df['xrealip']:
    if i.count('.') != 3:
        ip_df.drop(ip_df[ip_df['xrealip'] == i].index, axis = 0, inplace=True)
for j in df2['xrealip']:
    if j.count('.') != 3:
        df2.drop(df2[df2['xrealip'] == j].index, axis = 0, inplace=True)


# Split IP addresses into 4 octaves
ip_df.loc[:, 'oct1'] = ip_df['xrealip'].apply(lambda x: x.split(".")[0])
ip_df.loc[:, 'oct2'] = ip_df['xrealip'].apply(lambda x: x.split(".")[1])
ip_df.loc[:, 'oct3'] = ip_df['xrealip'].apply(lambda x: x.split(".")[2])
ip_df.loc[:, 'oct4'] = ip_df['xrealip'].apply(lambda x: x.split(".")[3])

ip_df = ip_df.reset_index()
ip_df = ip_df.drop(['index'], axis = 1)


# Covert binary number to decimal
oct1_dcm = []
for i in ip_df['oct1']:
    temp = int(i)*16777216
    oct1_dcm.append(temp)

oct2_dcm = []
for i in ip_df['oct2']:
    temp = int(i)*65536
    oct2_dcm.append(temp)

oct3_dcm = []
for i in ip_df['oct3']:
    temp = int(i)*256
    oct3_dcm.append(temp)

oct4_dcm = []
for i in ip_df['oct4']:
    temp = int(i)
    oct4_dcm.append(temp)

ip_df['oct1_dcm'] = oct1_dcm
ip_df['oct2_dcm'] = oct2_dcm
ip_df['oct3_dcm'] = oct3_dcm
ip_df['oct4_dcm'] = oct4_dcm

# Merge two files
ip_df = pd.merge(ip_df, df2, on = 'xrealip', how = 'inner')

# Create matrix for K-means clustering
X_matrix_ip = np.array(ip_df[['oct1_dcm', 'oct2_dcm', 'oct3_dcm', 'oct4_dcm']])


#apply PCA to reduce the dimensions for cluster visualization
pcas = PCA(n_components=4)
pcas.fit(X_matrix_ip)


pca_ip = PCA(n_components=2)
pcas = pca_ip.fit_transform(X_matrix_ip)

pcas1=[]
pcas2=[]

for i in pcas:
    pcas1.append(i[0])
    pcas2.append(i[1])


ip_df['pca1'] = pcas1
ip_df['pca2'] = pcas2


## k-mean clustering
ip_df2 = ip_df.copy()


# Elbow Method
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
  
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X_matrix_ip)
    kmeanModel.fit(X_matrix_ip)
  
    distortions.append(sum(np.min(cdist(X_matrix_ip, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X_matrix_ip.shape[0])
    inertias.append(kmeanModel.inertia_)
  
    mapping1[k] = sum(np.min(cdist(X_matrix_ip, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X_matrix_ip.shape[0]
    mapping2[k] = kmeanModel.inertia_


# Using the different values of Distortion
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


# Using the different values of Inertia
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()


# Select the number of clusters to continue
kclusters= int(input('Enter the number of clusters: '))

kms = KMeans(n_clusters=kclusters,n_init=200,random_state=0)

kms.fit_predict(X_matrix_ip)

ip_df2['kcluster']=kms.labels_.tolist()

C_matrix=kms.cluster_centers_

pca_centers = pca_ip.transform(C_matrix)

centers_df = pd.DataFrame(pca_centers, columns=['pca1', 'pca2'])

# Print the centers of all clusters in PCA values
print(centers_df)
print()
print(ip_df2)


dfkms0=ip_df2[ip_df2.kcluster==0]
dfkms1=ip_df2[ip_df2.kcluster==1]
dfkms2=ip_df2[ip_df2.kcluster==2]
# dfkms3=ip_df2[ip_df2.kcluster==3]


# Plot the graph of PCA
plt.rcParams["figure.figsize"] = (16,8)
plt.scatter(dfkms0['pca1'],dfkms0['pca2'],s=10,label="cl 0")
plt.scatter(dfkms1['pca1'],dfkms1['pca2'],s=10,marker="<",label="cl 1")
plt.scatter(dfkms2['pca1'],dfkms2['pca2'],s=10,marker="^",label="cl 2")
# plt.scatter(dfkms3['pca1'],dfkms3['pca2'],s=10,marker=">",label="cl 3")

plt.scatter(centers_df['pca1'], centers_df['pca2'],s=30,marker="D",label="Center", color='Black')

plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)


x_axis = pd.concat([dfkms0['pca1'],dfkms1['pca1'],dfkms2['pca1']], axis = 0)
x_axis = x_axis.reset_index()
del x_axis['index']

y_axis = pd.concat([dfkms0['pca2'],dfkms1['pca2'],dfkms2['pca2']], axis = 0)
y_axis = y_axis.reset_index()
del y_axis['index']


# Annotate the graph (both IP addresses and IP usage for 14 days)
ip_list = pd.concat([dfkms0['xrealip'],dfkms1['xrealip'],dfkms2['xrealip']], axis = 0)
ip_total_list = pd.concat([dfkms0['ip_total'],dfkms1['ip_total'],dfkms2['ip_total']], axis = 0)

txt_list = zip(ip_list, ip_total_list)

for i, txt in enumerate(txt_list):
    plt.annotate(txt, (x_axis.iloc[i].values[0], y_axis.iloc[i].values[0]))


plt.show()


# View clusters separately
print()
print(dfkms0)
print()
print(dfkms1)
print()
print(dfkms2)
print()
# print(dfkms3)


# Calculate the distance of the points from the centroids
def distance_from_center(oct1, oct2, oct3, oct4, label):
    '''
    Calculate the Euclidean distance between a data point and the center of its cluster.
    :param float income: the standardized income of the data point 
    :param float age: the standardized age of the data point 
    :param int label: the label of the cluster
    :rtype: float
    :return: The resulting Euclidean distance  
    '''
    center_oct1 =  kms.cluster_centers_[label,0]
    center_oct2 =  kms.cluster_centers_[label,1]
    center_oct3 =  kms.cluster_centers_[label,2]
    center_oct4 =  kms.cluster_centers_[label,3]
    distance = np.sqrt((oct1 - center_oct1) ** 2 + (oct2 - center_oct2) ** 2 + (oct3 - center_oct3) ** 2 + (oct4 - center_oct4) ** 2)
    distance =  preprocessing.scale(distance)
    return np.round(distance, 3)

ip_df2['label'] = kms.labels_
ip_df2['distance'] = distance_from_center(ip_df2.oct1_dcm, ip_df2.oct2_dcm, ip_df2.oct3_dcm, ip_df2.oct4_dcm, ip_df2.label)
print(ip_df2)
print()

# Select the percentile of distance and/or ip usage to exclude outliers
percentile_dist = np.percentile(ip_df2['distance'], 95)
percentile_ip_usage = np.percentile(pd.to_numeric(ip_total_list), 10)


ip_df_new = ip_df2.loc[(ip_df2['distance'] <= percentile_dist)]
print(f'percentile of distance = {percentile_dist}')
print(f'percentile of ip usage = {percentile_ip_usage}')
print()
print(ip_df_new)




