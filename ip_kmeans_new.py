import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import os
import ipaddress

os.environ['OMP_NUM_THREADS'] = '11'


df_file = "ip_one_account.csv"

df = pd.read_csv(df_file, \
                 dtype=object)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

ip_df = df['xrealip'].copy()
ip_df = ip_df.reset_index()
del ip_df['index']
ip_df = ip_df.dropna().copy()
ip_df = ip_df.drop_duplicates().copy()
# print(ip_df.head(10))


ip_df.loc[:, 'oct1'] = ip_df['xrealip'].apply(lambda x: x.split(".")[0])
ip_df.loc[:, 'oct2'] = ip_df['xrealip'].apply(lambda x: x.split(".")[1])
ip_df.loc[:, 'oct3'] = ip_df['xrealip'].apply(lambda x: x.split(".")[2])
ip_df.loc[:, 'oct4'] = ip_df['xrealip'].apply(lambda x: x.split(".")[3])

ip_df = ip_df.reset_index()
ip_df = ip_df.drop(['index'], axis = 1)
# print(ip_df.head(10))

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

# print(ip_df.head(10))


X_matrix_ip = np.array(ip_df[['oct1_dcm', 'oct2_dcm', 'oct3_dcm', 'oct4_dcm']])
# print(X_matrix_ip[0:5])

# kmeans=KMeans(n_clusters=3)
# a = kmeans.fit(X_matrix_ip)
# centroids=kmeans.cluster_centers_
# labels=kmeans.labels_
# print("centroids",centroids)
# print ("labels",labels)


kclusters=3
kms = KMeans(n_clusters=kclusters,n_init= 200, random_state=0)
# 
kms.fit_predict(X_matrix_ip)

ip_df['kcluster']=kms.labels_.tolist()

C_matrix=kms.cluster_centers_
print(C_matrix)

dfkms0=ip_df[ip_df.kcluster==0]
dfkms1=ip_df[ip_df.kcluster==1]
dfkms2=ip_df[ip_df.kcluster==2]
# dfkms3=ip_df[ip_df.kcluster==3]

print()
print(dfkms0)
print(dfkms1)
print(dfkms2)
# print(dfkms3)







#apply PCA to reduce the dimensions for cluster visualization
pcas = PCA(n_components=4)
pcas.fit(X_matrix_ip)
# print(pcas.explained_variance_ratio_)

pca_ip = PCA(n_components=2)
pcas = pca_ip.fit_transform(X_matrix_ip)

pcas1=[]
pcas2=[]

for i in pcas:
    pcas1.append(i[0])
    pcas2.append(i[1])

# print(len(pcas1))
# print(len(pcas2))

ip_df['pca1'] = pcas1
ip_df['pca2'] = pcas2
# print(ip_df[:2])

C_matrix=kms.cluster_centers_

pca_centers = pca_ip.transform(C_matrix)

centers_df = pd.DataFrame(pca_centers, columns=['pca1', 'pca2'])


ip_df_copy = ip_df.copy()

plt.rcParams["figure.figsize"] = (16,8)
plt.scatter(dfkms0['pca1'],dfkms0['pca2'],s=10,label="cl 0")
plt.scatter(dfkms1['pca1'],dfkms1['pca2'],s=10,marker="<",label="cl 1")
plt.scatter(dfkms2['pca1'],dfkms2['pca2'],s=10,marker="^",label="cl 2")

plt.scatter(centers_df['pca1'], centers_df['pca2'],s=30,marker="D",label="Center", color='Black')

plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
for i_x, i_y in zip(dfkms0['pca1'], dfkms0['pca2']):
    plt.text(i_x, i_y, '({:.3f}, {:.3f})'.format(i_x, i_y))
for i_x, i_y in zip(dfkms1['pca1'], dfkms1['pca2']):
    plt.text(i_x, i_y, '({:.3f}, {:.3f})'.format(i_x, i_y))
for i_x, i_y in zip(dfkms2['pca1'], dfkms2['pca2']):
    plt.text(i_x, i_y, '({:.3f}, {:.3f})'.format(i_x, i_y))
plt.show()