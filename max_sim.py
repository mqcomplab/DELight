import numpy as np

fps_train_idxs_1 = np.load('../data/fps_train_idxs_1.npy', mmap_mode='r')
fp_train_idxs = np.load('../data/fps_train_idxs.npy', mmap_mode='r')
fps_train_idxs_0 = np.load('../data/fps_train_idxs_0.npy', mmap_mode='r')
fp = np.load('../../fps_combined.npy', mmap_mode='r')
n_train_1 = len(fps_train_idxs_1)


def get_valid_clusters(file_path):
    threshold_cluster = []
    with open(file_path, 'r') as f:
        for line in f:
            cleaned_line = line.strip().replace('[', '').replace(']', '').replace(',', '')
            threshold_cluster.append(list(map(int, cleaned_line.split())))

    threshold_cluster.sort(key=len, reverse=True)
    return threshold_cluster


file_paths = ['../cluster/fps_0.9_big_clus.txt', 
              '../cluster/fps_0.8_big_clus.txt', 
              '../cluster/fps_0.7_big_clus.txt',
              '../cluster/fps_0.6_big_clus.txt',
              '../cluster/fps_0.5_big_clus.txt',
              '../cluster/fps_0.4_clusters.txt'
              ]

cluster_list = []
for file_path in file_paths:
    threshold_cluster = get_valid_clusters(file_path)
    cluster_list.append(threshold_cluster)
    print(file_path)

inactive_idxs = []
fps_train_idxs_1= set(fps_train_idxs_1)
pops = []

for j, threshold in enumerate(cluster_list):
    for i, cluster in enumerate(reversed(threshold)):
        cluster_set = set(cluster)
        intersection = cluster_set & fps_train_idxs_1
        if intersection:
            pop = len(cluster) / n_train_1
            if len(cluster) <= 19:
                pops.append(round(pop, 6))
                print(f"cluster is in {j}")
                print(cluster[:5])
                inactive_idxs.extend([idx for idx in cluster if idx not in intersection])
                
                if len(inactive_idxs) >= n_train_1:
                    break
    if len(inactive_idxs) >= n_train_1:
        break

with open('log.txt', 'w') as f:
    f.write(f"Number of inactive mols: {len(inactive_idxs)}\n")
    f.write(f"Number of fps_train_idxs_1: {n_train_1}\n")
    f.write(f"Percentage of inactive mols: {len(inactive_idxs) / n_train_1:.2%}\n")
    f.write(f'Population of the top 20 clusters: {pops}\n')

with open('inactive_idx.txt', 'w') as f:
    for idx in inactive_idxs:
        f.write(f"{idx}\n")