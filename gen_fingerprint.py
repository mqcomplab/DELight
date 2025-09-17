import pandas as pd
import numpy as np

data_file = "../data/WDR91.tsv"

# Define the size of chunks (you can adjust this depending on your system's memory)
chunk_size = 100000  # Process 100,000 rows at a time

# Initialize empty lists to store the results
fingerprints_0 = []
fingerprints_1 = []

# Read the file in chunks to avoid loading everything into memory
for chunk in pd.read_csv(data_file, sep="\t", header=None, chunksize=chunk_size):
    # Extract labels and fingerprints for this chunk
    labels = chunk[2]  # Column 3 (index 2)
    fingerprints = chunk[10]  # Column 11 (index 10)

    # Append to the corresponding lists based on labels
    fingerprints_0.extend(fingerprints[labels == 0].tolist())  # Add fingerprints where label is 0
    fingerprints_1.extend(fingerprints[labels == 1].tolist())  # Add fingerprints where label is 1

# Convert the lists to numpy arrays after processing the entire file
fingerprints_0 = np.array(fingerprints_0)
fingerprints_1 = np.array(fingerprints_1)

# Save the arrays to disk
np.save('fingerprints_0.npy', fingerprints_0)
np.save('fingerprints_1.npy', fingerprints_1)

print("Arrays saved successfully!")
fps_1 = np.load('fingerprints_1.npy')
fps_0 = np.load('fingerprints_0.npy')
fps_1 = np.array([np.fromstring(fp, sep=',') for fp in fps_1])
np.save('fps_1.npy', fps_1)
fps_0 = np.array([np.fromstring(fp, sep=',') for fp in fps_0])
np.save('fps_0.npy', fps_0)
fps_combined = np.concatenate((fps_1, fps_0), axis=0)
np.save('fps_combined.npy', fps_combined)