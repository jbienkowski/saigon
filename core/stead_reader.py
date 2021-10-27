import h5py
import pandas as pd
import numpy as np
import random

from .entities import SteadDataObject


class SteadReader:
    def __init__(self, cfg):
        self._cfg = cfg

    def get_processed_data(self, idx_start, idx_end, idx_slice):
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        with h5py.File(self._cfg["stead_path_db_processed"], "r") as f:
            x_train = f["data"][idx_start:idx_slice]
            y_train = f["labels"][idx_start:idx_slice]
            x_test = f["data"][idx_slice:idx_end]
            y_test = f["labels"][idx_slice:idx_end]
            return (x_train, y_train, x_test, y_test)

    def get_data_by_evi(self, evi):
        dtfl = h5py.File(self._cfg["stead_path_db"], "r")
        dataset = dtfl.get(f"data/{evi}")
        do = self.to_dataobject(dataset)
        return do

    def get_event_data(self, idx_start, idx_end):
        streams = []
        df = pd.read_csv(self._cfg["stead_path_csv"])
        df = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 75)
            & (df.source_magnitude > 1.5)
        ]
        ev_list = df["trace_name"].to_list()[idx_start:idx_end]

        dtfl = h5py.File(self._cfg["stead_path_db"], "r")

        for _, evi in enumerate(ev_list):
            dataset = dtfl.get(f"data/{evi}")
            do = self.to_dataobject(dataset)
            streams.append(do)

        return streams

    def get_noise_data(self, idx_start, idx_end):
        streams = []
        df = pd.read_csv(self._cfg["stead_path_csv"])
        df = df[(df.trace_category == "noise")]
        ev_list = df["trace_name"].to_list()[idx_start:idx_end]

        dtfl = h5py.File(self._cfg["stead_path_db"], "r")

        for _, evi in enumerate(ev_list):
            dataset = dtfl.get(f"data/{evi}")
            do = self.to_dataobject(dataset)
            streams.append(do)

        return streams

    def prepare_data(self):
        keys = []

        df = pd.read_csv(self._cfg["stead_path_csv"])

        df_eq = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 75)
            & (df.source_magnitude > 1.5)
        ]
        eq_list = df_eq["trace_name"].to_list()[:50000]

        df_an = df[(df.trace_category == "noise")]
        an_list = df_an["trace_name"].to_list()[:50000]

        for eq in eq_list:
            keys.append(eq)

        for an in an_list:
            keys.append(an)

        random.shuffle(keys)
        len_keys = len(keys)

        # Init the HDF5 file
        with h5py.File(self._cfg["stead_path_db_processed"], "a") as f:
            f.create_dataset("keys", shape=(len_keys,), dtype="S50")
            f.create_dataset("data", shape=(len_keys, 3, 6000), dtype="float32")
            f.create_dataset("labels", shape=(len_keys,), dtype="i4")

        for idx, key in enumerate(keys):
            if idx % 100 == 0:
                print(f"{idx}/{len_keys}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_path_db_processed"], "a") as f:
                f["keys"][idx] = bytes(key, encoding="utf-8")
                f["data"][idx] = do.get_components()
                f["labels"][idx] = 1 if do.trace_name.lower().endswith("ev") else 0

    def to_dataobject(self, obj) -> SteadDataObject:
        do = SteadDataObject()
        setattr(do, "data", np.array(obj))

        for a in obj.attrs:
            setattr(do, a, obj.attrs[a])

        return do
