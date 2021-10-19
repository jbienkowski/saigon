import h5py
import random
import numpy as np
from numpy import savez_compressed
from .entities import DataObject


class Hd5Reader:
    def __init__(self, path: str):
        self._path = path

    def get_subset_of_processed_data(self, size):
        keys = None
        data = None
        labels = None
        with h5py.File(self._path, "r") as f:
            keys = f["keys"][:size]
            data = f["data"][:size]
            labels = f["labels"][:size]
            return (keys, data, labels)

    def get_data(self, idx_start, idx_end, idx_slice):
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        with h5py.File(self._path, "r") as f:
            x_train = f["data"][idx_start:idx_slice]
            y_train = f["labels"][idx_start:idx_slice]
            x_test = f["data"][idx_slice:idx_end]
            y_test = f["labels"][idx_slice:idx_end]
            return (x_train, y_train, x_test, y_test)

    def prepare_data(self):
        file_name = "LEN-DB-processed.hdf5"
        keys = []
        with h5py.File(self._path, "r") as f:
            [keys.append(f"AN/{an}") for an in f["AN"].keys()]
            [keys.append(f"EQ/{eq}") for eq in f["EQ"].keys()]
        random.shuffle(keys)

        len_keys = len(keys)

        # Init the HDF5 file
        with h5py.File(f"data/{file_name}", "a") as f:
            f.create_dataset("keys", shape=(len_keys,), dtype="S50")
            f.create_dataset("data", shape=(len_keys, 1620), dtype="float64")
            f.create_dataset("labels", shape=(len_keys,), dtype="i8")

        with h5py.File(self._path, "r") as f1:
            for idx, key in enumerate(keys):
                if idx % 100 == 0:
                    print(f"{idx}/{len_keys}")
                with h5py.File(f"data/{file_name}", "a") as f2:
                    f2["keys"][idx] = bytes(key, encoding="utf-8")
                    f2["data"][idx] = np.array(f1.get(key)).reshape(1620)
                    f2["labels"][idx] = 0 if key.startswith("AN") else 1

    def get_random_object(self, type: str) -> DataObject:
        with h5py.File(self._path, "r") as f:
            id = list(f[type.upper()])[random.randint(0, 6e5)]
            obj = f.get(f"{type.upper()}/{id}")
            do = DataObject()
            do.type = type
            do.id = id
            do.data = np.array(obj)

            for a in obj.attrs:
                setattr(do, a, obj.attrs[a])

            return do

    def find_dataobject(self, id: str) -> DataObject:
        with h5py.File(self._path, "r") as f:
            obj = f.get(id)
            do = DataObject()
            do.type = id.split("/")[0]
            do.id = id.split("/")[1]
            do.data = np.array(obj)

            for a in obj.attrs:
                setattr(do, a, obj.attrs[a])

            return do
