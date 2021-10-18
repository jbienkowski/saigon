import h5py
import random
import numpy as np
from numpy import savez_compressed
from .entities import DataObject


class Hd5Reader:
    def __init__(self, path: str):
        self._path = path

    def prepare_data(self):
        file_name = "processed_data.hdf5"
        keys = []
        with h5py.File(self._path, "r") as f:
            [keys.append(f"AN/{an}") for an in f["AN"].keys()]
            [keys.append(f"EQ/{eq}") for eq in f["EQ"].keys()]
        random.shuffle(keys)

        len_keys = len(keys)

        try:
            with h5py.File(file_name, "a") as f:
                f.create_dataset("keys", shape=(len_keys,), dtype="S50")
                f.create_dataset("data", shape=(len_keys, 1620))
                f.create_dataset("labels", shape=(len_keys,), dtype="i1")
        except Exception as ex:
            print(str(ex))

        with h5py.File(self._path, "r") as f1:
            for idx, key in enumerate(keys):
                if idx % 100 == 0:
                    print(f"{idx}/{len_keys}")
                with h5py.File(file_name, "a") as f2:
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
