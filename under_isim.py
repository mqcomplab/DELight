import numpy as np

# Load fingerprint matrix and index files
fp = np.load('../../fps_combined.npy', mmap_mode='r')
act_fp_idxs = np.load('../data/fps_train_idxs_1.npy', mmap_mode='r')
neg_fp_idxs = np.load('../data/fps_train_idxs_0.npy', mmap_mode='r')

# Extract active and inactive fingerprints
act_fps = fp[act_fp_idxs]
neg_fps = fp[neg_fp_idxs]
n_acts = len(act_fps)

# Compute tanimoto-like similarity
act_sum = np.sum(act_fps, axis=0)
c_totals = act_sum + fp  # all fingerprints included
a = np.sum(c_totals * (c_totals - 1) / 2, axis=1)
off_coincidences = n_acts + 1 - c_totals
total_dis = np.sum(off_coincidences * c_totals, axis=1)
tanis = a / (a + total_dis)

# Get index range for inactives in global fp array
inactive_start_idx = n_acts
inactive_end_idx = inactive_start_idx + len(neg_fps)

# Sort by tanis: most and least similar
isim_max = np.argsort(-tanis)
isim_min = np.argsort(tanis)

# Collect most similar inactives
neg_isim_max = []
for j in isim_max:
    if inactive_start_idx <= j < inactive_end_idx:
        neg_isim_max.append(j)
    if len(neg_isim_max) >= n_acts:
        break

# Collect least similar inactives
neg_isim_min = []
for j in isim_min:
    if inactive_start_idx <= j < inactive_end_idx:
        neg_isim_min.append(j)
    if len(neg_isim_min) >= n_acts:
        break

# Convert global indices back to original index list values
neg_fps_isim_max = [idx for idx in neg_fp_idxs if idx in neg_isim_max]
neg_fps_isim_min = [idx for idx in neg_fp_idxs if idx in neg_isim_min]

# Save to file
np.savetxt("neg_fps_isim_max.csv", neg_fps_isim_max, fmt='%i')
np.savetxt("neg_fps_isim_min.csv", neg_fps_isim_min, fmt='%i')
