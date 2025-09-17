import numpy as np

undersampling_file = '../max_sim/inactive_idx.txt'


X_all = np.load('../../fps_combined.npy', mmap_mode='r')
X_test = np.load('../data/fps_test_idxs.npy', mmap_mode='r')

# loaded to get the number of samples
n_rows = len(X_all)

# Testing labels
active_test = np.load('../data/fps_test_idxs_1.npy', mmap_mode='r')
y_test = np.zeros(n_rows, dtype=int)
y_test[active_test] = 1 # Set active samples to 1
y_test = y_test[X_test] # only keep testing indices.

# Checking that the number of active samples is correct
active_count = np.sum(y_test)
assert active_count == len(active_test), f"Expected {len(active_test)} active samples, but found {active_count}."

# Save the labels to files
print('y_test shape:', y_test.shape)
np.save('y_test.npy', y_test)

# Training labels
y_train = np.zeros(n_rows, dtype=int)
to_keep = np.loadtxt(undersampling_file, dtype=int) # Load inactive indices
active_train = np.load('../data/fps_train_idxs_1.npy', mmap_mode='r')
final_indices = np.concatenate([to_keep, active_train])

# Set correct labels
y_train[to_keep] = 0
y_train[active_train] = 1

# Subset y_train to only the desired indices
y_train = y_train[final_indices]

# Save the relevant lines from X_all
X_train = X_all[final_indices]

# Checking that the number of active samples is correct
active_count = np.sum(y_train)
assert active_count == len(active_train), f"Expected {len(active_train)} active samples, but found {active_count}."

# Save the relevant lines from X_all
np.save('fps_undersampled.npy', X_train)
print('X_train', X_train.shape)

# Save the labels to files
print('y_train shape:', y_train.shape)
np.save('y_train.npy', y_train)