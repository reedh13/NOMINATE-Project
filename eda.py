import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("data/HSall_members.csv")

# Remove entries before year 1901
df = df[df.congress >= 57]

# Creat DFs House members, Senators, and all Congress
df_house = df[df.chamber == 'House']
df_sen = df[df.chamber == 'Senate']
df_congress = df[(df.chamber == 'House') | (df.chamber == 'Senate')]

def plot_per_congress(df, show=False, save=None):
    blue = 'dodgerblue'
    red = 'red'
    green = 'green'
    alpha = 0.4

    blue_patch = mpatches.Patch(color=blue, label='Democrat')
    red_patch = mpatches.Patch(color=red, label='Republican')
    green_patch = mpatches.Patch(color=green, label='Other')

    start_yr = 1901
    # Create a plot for each Congress
    for i in range(57, 117):
        df_temp = df[df.congress == i]
        unique_parties = list(set(df_temp['party_code'].tolist()))

        for party in unique_parties:
            if party == 100:
                plt.scatter(df_temp[df_temp.party_code == party]['nokken_poole_dim1'].tolist(), df_temp[df_temp.party_code == party]['nokken_poole_dim2'].tolist(), alpha=alpha, color=blue)
            elif party == 200:
                plt.scatter(df_temp[df_temp.party_code == party]['nokken_poole_dim1'].tolist(), df_temp[df_temp.party_code == party]['nokken_poole_dim2'].tolist(), alpha=alpha, color=red)
            else:
                plt.scatter(df_temp[df_temp.party_code == party]['nokken_poole_dim1'].tolist(), df_temp[df_temp.party_code == party]['nokken_poole_dim2'].tolist(), alpha=alpha, color=green)

        title = f'DW-NOMINATE Values for US {save} {i} ({start_yr} - {start_yr+2})'
        plt.title(title)
        plt.xlim([-1.25, 1.25])
        plt.ylim([-1.25, 1.25])
        plt.xlabel('DW-Nominate D1')
        plt.ylabel('DW-Nominate D2')
        plt.legend(loc='upper right', 
            ncol=3, 
            borderaxespad=0.2, 
            handles=[blue_patch, red_patch, green_patch])
        if save != None:
            plt.savefig('images/' + save + str(i) + '.png', format='png', dpi=300)
        if show:
            plt.show()
        plt.close()

        start_yr += 2

#plot_per_congress(df_sen, show=False, save='Senate')
#plot_per_congress(df_house, show=False, save='House')
#plot_per_congress(df_congress, show=False, save='Congress')


#print(df_congress[df_congress.bioname == 'JONES, Walter Beaman, Jr.'])

df_heir = df_congress[(df_congress.congress == 116) & (df_congress.bioname != 'JONES, Walter Beaman, Jr.')]
X = df_heir[['nokken_poole_dim1', 'nokken_poole_dim2']].to_numpy()

# Scipy Method
# plt.figure(figsize=(20, 7))
# dendrogram(linkage(vals),
#             orientation='top',
#             labels=df_heir.bioname.tolist(),
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()


# Sklearn method
def sk_heir(X, n, method):
    hc = AgglomerativeClustering(n_clusters = n, affinity = 'euclidean', linkage = method)
    y_hc = hc.fit_predict(X)

    plt.scatter(X[:,0],X[:,1], c=y_hc, cmap='Dark2', alpha=0.4)
    # plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'green', label = 'Cluster 1', alpha=0.4)
    # plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'purple', label = 'Cluster 2', alpha=0.4)
    plt.title(f'Agglomerative hierarchical clustering: {method} linkage, n={n}')
    plt.xlabel('DW-Nominate D1')
    plt.ylabel('DW-Nominate D2')
    plt.show()

sk_heir(X, 6, 'single')