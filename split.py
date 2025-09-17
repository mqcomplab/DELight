"""
train_idxs: idxs of the data points selected for training.
test_idxs: idxs of the data points selected for testing.
fps_1_idx: idxs of the data points belonging to the fps_1 category.
fps_0_idx: idxs of the data points belonging to the fps_0 category.
train_idxs_1: idxs of the data points selected for training that belong to the fps_1 category.
train_idxs_0: idxs of the data points selected for training that belong to the fps_0 category.
test_idxs_1: idxs of the data points selected for testing that belong to the fps_1 category.
test_idxs_0: idxs of the data points selected for testing that belong to the fps_0 category.
All idxs are based on the original fps_combined.npy file.
"""

import numpy as np

training_percentage = 0.8

fps_1 = np.load('../fps_1.npy', mmap_mode='r')
fps_0 = np.load('../fps_0.npy', mmap_mode='r')
fps_1_idx = np.arange(len(fps_1))
fps_0_idx = np.arange(len(fps_1), len(fps_1) + len(fps_0))
fps = np.load('../fps_combined.npy', mmap_mode='r')

# take 80% of the data for training and 20% for testing
train_size = int(len(fps) * training_percentage)
test_size = len(fps) - train_size
train_idxs = np.random.choice(len(fps), size=train_size, replace=False)
test_idxs = np.setdiff1d(np.arange(len(fps)), train_idxs)
test_idxs_1 = np.intersect1d(test_idxs, fps_1_idx)

# Optional step: check if test set has between 3% to 12% of the data from fps_1
# if 0.03 < len(test_idxs_1) / test_size < 0.12:
#     with open('log.txt', 'a') as f:
#         f.write(f"Test set in fp1 percentage: {len(test_idxs_1) / test_size:.2%}\n")
#         f.write("Test set has between 3% to 12% of the data from fps_1\n")
    
# else:
#     with open('log.txt', 'a') as f:
#         f.write(f"Test set in fp1 percentage: {len(test_idxs_1) / test_size:.2%}\n")
#         f.write("Test set does NOT have between 3% to 12% of the data from fps_1\n") 

test_idxs_0 = np.intersect1d(test_idxs, fps_0_idx)
train_fps = fps[train_idxs]
test_fps = fps[test_idxs]

train_idxs_1 = np.intersect1d(train_idxs, fps_1_idx)
train_idxs_0 = np.intersect1d(train_idxs, fps_0_idx)
np.save('fps_train.npy', train_fps)
np.save('fps_test.npy', test_fps)
np.save('fps_train_idxs.npy', train_idxs)
np.save('fps_test_idxs.npy', test_idxs)
np.save('fps_train_idxs_1.npy', train_idxs_1)
np.save('fps_train_idxs_0.npy', train_idxs_0)
np.save('fps_test_idxs_1.npy', test_idxs_1)
np.save('fps_test_idxs_0.npy', test_idxs_0)