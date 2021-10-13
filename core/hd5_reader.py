import logging
import h5py
import numpy as np
from .models import DataObject


class Hd5Reader:
    def __init__(self, path: str):
        self._path = path

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
