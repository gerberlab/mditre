import pickle

# Load MITRE dataset object
with open('./datasets/david_reference_dataset_object.pickle', 'rb') as f:
    pickle_dataset = pickle.load(f)

# Convert MITRE dataset object into dictionary for ease of use
dict_dataset = {k: getattr(pickle_dataset, k) for k in vars(pickle_dataset).keys()}
dict_dataset['T'] = [x.astype('int') for x in dict_dataset['T']]

# Save the new dataset
with open('./datasets/david_agg_filtered.pickle', 'wb') as f:
    pickle.dump(dict_dataset, f)