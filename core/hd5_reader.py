import h5py
import random
import numpy as np
from numpy import savez_compressed
from .entities import DataObject


class Hd5Reader:
    def __init__(self, path: str):
        self._path = path

    def get_subset_of_processed_data(self, train_size, test_size, valid_size):
        total_size = train_size + test_size + valid_size
        chunk = int(total_size / 2)
        eqs = []
        ans = []
        with h5py.File(self._path, "r") as f:

            keys_eq = list(f["EQ"])[:chunk]
            for key_eq in keys_eq:
                eq = f.get(f"EQ/{key_eq}")
                do = self.to_dataobject(eq)
                eqs.append(do)

            keys_ans = list(f["AN"])[:chunk]
            for key_an in keys_ans:
                an = f.get(f"AN/{key_an}")
                do = self.to_dataobject(an)
                ans.append(do)

        combined = eqs + ans
        random.shuffle(combined)
        return (
            combined[:train_size],
            combined[train_size : train_size + test_size],
            combined[train_size + test_size : train_size + test_size + valid_size],
        )

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
            f.create_dataset("data", shape=(len_keys, 3, 540), dtype="float64")
            f.create_dataset("labels", shape=(len_keys,), dtype="i4")

        with h5py.File(self._path, "r") as f1:
            for idx, key in enumerate(keys):
                if idx % 100 == 0:
                    print(f"{idx}/{len_keys}")
                with h5py.File(f"data/{file_name}", "a") as f2:
                    f2["keys"][idx] = bytes(key, encoding="utf-8")
                    f2["data"][idx] = np.array(f1.get(key))
                    f2["labels"][idx] = 1 if key.split("/")[0].lower() == "eq" else 0

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

    def to_dataobject(self, obj) -> DataObject:
        do = DataObject()
        do.type = obj.name.split("/")[1]
        do.id = obj.name.split("/")[2]
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
