import h5py
import random
import numpy as np
from .models import DataObject


class Hd5Reader:
    def __init__(self, path: str):
        self._path = path

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
