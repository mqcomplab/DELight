import numpy as np
import bitbirch.bitbirch as bb


train_fps = np.load('../data/fps_train.npy', mmap_mode='r')
fps_train_idxs_1 = np.load('../data/fps_train_idxs_1.npy', mmap_mode='r')
train_og_idxs = np.load('../data/fps_train_idxs.npy', mmap_mode='r')
og_fp = np.load('../../fps_combined.npy', mmap_mode='r')
"""
with open('fps_0.9_clusters.txt', 'r') as f:
    cluster_list = []
    for line in f:
        cleaned_line = line.strip().replace('[', '').replace(']', '').replace(',', '')
        cluster_list.append(list(map(int, cleaned_line.split())))
        
low_density = [cluster for cluster in cluster_list if len(cluster) <= 10]
fps_low_d_fp_idxs = []
low_d_fp_idxs = []

for cluster in low_density:
    fps_low_d_fp_idxs.append(og_fp[cluster])
    low_d_fp_idxs.append(cluster)

fps_low_d_fp_idxs = np.vstack(fps_low_d_fp_idxs)
fps_idx_flat = np.hstack(low_d_fp_idxs)
"""
def BFs_reclustering(fps, init_threshold, second_threshold, second_tolerance, branching_factor=50):
    bb.set_merge('diameter')
    brc = bb.BitBirch(branching_factor=branching_factor, threshold=init_threshold)
    brc.fit(fps, singly=True)

    # Extract the BFs for the second clustering
    BFs, big = brc.prepare_data_BFs(fps)

    # Do the second clustering
    bb.set_merge('tolerance', tolerance=second_tolerance)
    brc = bb.BitBirch(branching_factor=branching_factor, threshold=second_threshold)
    brc.fit_BFs(BFs) # Note that we fit the BFs, not the fps
    brc.fit_BFs(big) # Fit the big cluster as well but in individual BFs
    cluster_list = brc.get_cluster_mol_ids()
    fps_low_d_fp_idxs, fps_idx_flat = write_clusters_0(cluster_list, second_threshold, fps_train_idxs_1)
    return fps_low_d_fp_idxs, fps_idx_flat


def cluster(fps_low_d_fp_idxs, branching_factor, threshold, fps_idx_flat):
    bb.set_merge('diameter')
    brc = bb.BitBirch(branching_factor=branching_factor, threshold=threshold)
    brc.fit(fps_low_d_fp_idxs)
    
    cluster_list = brc.get_cluster_mol_ids()
    fps_low_d_fp_idxs, fps_idx_flat = write_clusters_1(cluster_list, threshold, fps_train_idxs_1, fps_idx_flat)
    return fps_low_d_fp_idxs, fps_idx_flat


def write_clusters_0(cluster_list, threshold, fps_1_idx):
    fps_1_idx_set = set(fps_1_idx)
    bb_clusters = []
    big_clusters = []
    low_d_fp_idxs = []
    
    for cluster in cluster_list:
        og_cluster_idxs = [train_og_idxs[i] for i in cluster]
        bb_clusters.append(og_cluster_idxs)
        if len(cluster) > 10:
            big_clusters.append(og_cluster_idxs)
        else:
            low_d_fp_idxs.append(og_cluster_idxs)

    with open(f'fps_{threshold}_singletons.txt', 'w') as f_singletons:
        for cluster in low_d_fp_idxs:
            if len(cluster) == 1 and cluster[0] in fps_1_idx_set:
                f_singletons.write(f"{cluster}\n")

    with open(f'fps_{threshold}_clusters.txt', 'w') as f_clusters:
        for cluster in bb_clusters:
            f_clusters.write(f"{cluster}\n")
    with open(f'fps_{threshold}_big_clus.txt', 'w') as f_big_clusters:
        for cluster in big_clusters:
            f_big_clusters.write(f"{cluster}\n")
    
        
    fps_low_d_fp_idxs = []
    for cluster in low_d_fp_idxs:
        fps_low_d_fp_idxs.append(og_fp[cluster])
    fps_low_d_fp_idxs = np.vstack(fps_low_d_fp_idxs)
    fps_idx_flat = np.hstack(low_d_fp_idxs)
    return fps_low_d_fp_idxs, fps_idx_flat

def write_clusters_1(cluster_list, threshold, fps_1_idx, fps_idx_flat):
    fps_1_idx_set = set(fps_1_idx)
    bb_clusters = []
    big_clusters = []
    low_d_fp_idxs = []
    
    for cluster in cluster_list:
        og_cluster_idxs = [fps_idx_flat[i] for i in cluster]
        bb_clusters.append(og_cluster_idxs)
        if len(cluster) > 10:
            big_clusters.append(og_cluster_idxs)
        else:
            low_d_fp_idxs.append(og_cluster_idxs)

    with open(f'fps_{threshold}_singletons.txt', 'w') as f_singletons:
        for cluster in low_d_fp_idxs:
            if len(cluster) == 1 and cluster[0] in fps_1_idx_set:
                f_singletons.write(f"{cluster}\n")

    with open(f'fps_{threshold}_clusters.txt', 'w') as f_clusters:
        for cluster in bb_clusters:
            f_clusters.write(f"{cluster}\n")
    with open(f'fps_{threshold}_big_clus.txt', 'w') as f_big_clusters:
        for cluster in big_clusters:
            f_big_clusters.write(f"{cluster}\n")
    
        
    fps_low_d_fp_idxs = []
    for cluster in low_d_fp_idxs:
        fps_low_d_fp_idxs.append(og_fp[cluster])
    fps_low_d_fp_idxs = np.vstack(fps_low_d_fp_idxs)
    fps_idx_flat = np.hstack(low_d_fp_idxs)
    return fps_low_d_fp_idxs, fps_idx_flat

fps_low_d_fp_idxs, fps_idx_flat = BFs_reclustering(train_fps, init_threshold=0.9, second_threshold=0.9, second_tolerance=0.05)
fps_low_d_fp_idxs, fps_idx_flat = cluster(fps_low_d_fp_idxs, branching_factor=50, threshold=0.8, fps_idx_flat=fps_idx_flat)
fps_low_d_fp_idxs, fps_idx_flat = cluster(fps_low_d_fp_idxs, branching_factor=50, threshold=0.7, fps_idx_flat=fps_idx_flat)
fps_low_d_fp_idxs, fps_idx_flat = cluster(fps_low_d_fp_idxs, branching_factor=50, threshold=0.6, fps_idx_flat=fps_idx_flat)
fps_low_d_fp_idxs, fps_idx_flat = cluster(fps_low_d_fp_idxs, branching_factor=50, threshold=0.5, fps_idx_flat=fps_idx_flat)
fps_low_d_fp_idxs, fps_idx_flat = cluster(fps_low_d_fp_idxs, branching_factor=50, threshold=0.4, fps_idx_flat=fps_idx_flat)
fps_low_d_fp_idxs, fps_idx_flat = cluster(fps_low_d_fp_idxs, branching_factor=50, threshold=0.3, fps_idx_flat=fps_idx_flat)
