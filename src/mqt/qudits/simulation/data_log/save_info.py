import os
import uuid
import getpass

import h5py
import numpy as np


def save_full_states(list_of_vectors, file_path=None, file_name=None):
    if file_name is None:
        file_name = "experiment_states.h5"

    if file_path is None:
        username = getpass.getuser()
        file_path = os.path.join(f"/home/{username}/Documents", file_name)
    else:
        file_path = os.path.join(file_path, file_name)

    size = list_of_vectors[0].shape

    # Generate random unique names for each vector
    vector_names = [str(uuid.uuid4()) for _ in range(len(list_of_vectors))]

    # Combine names and vectors into a list of dictionaries
    list_of_vectors = [{'name': name, 'data': vector} for name, vector in zip(vector_names, list_of_vectors)]

    # Open the HDF5 file in write mode
    with h5py.File(file_path, 'w') as hdf_file:
        # Create a table dataset within the file to store the vectors
        dtype = [('name', 'S36'), ('vector_data', np.complex128, size)]
        table_data = np.array(
                [(vector_info['name'].encode('utf-8'), vector_info['data']) for vector_info in list_of_vectors],
                dtype=dtype
        )

        hdf_file.create_dataset('vectors', data=table_data)


def save_shots(shots, file_path=None, file_name=None):
    if file_name is None:
        file_name = "experiment_shots.h5"

    if file_path is None:
        username = getpass.getuser()
        file_path = os.path.join(f"/home/{username}/Documents", file_name)
    else:
        file_path = os.path.join(file_path, file_name)

    indexes = [_ for _ in range(len(shots))]

    # Combine names and vectors into a list of dictionaries
    list_of_outcomes = [{'shot_nr': nr, 'outcome': outcome} for nr, outcome in zip(indexes, shots)]

    with h5py.File(file_path, 'w') as hdf_file:
        # Create a table dataset within the file to store the vectors
        dtype = [('nr', int), ('shot', int)]
        table_data = np.array([(shot_info['shot_nr'], shot_info['outcome']) for shot_info in list_of_outcomes], dtype = dtype)
        hdf_file.create_dataset('shots', data=table_data)
