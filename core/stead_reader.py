import h5py
import pandas as pd
import numpy as np
import random

from .stead_plotter import SteadPlotter
from .entities import SteadDataObject
from scipy.signal import stft, istft, resample

COMP_MAP = ["Z", "N", "E"]
COMP_QTY = 3


class SteadReader:
    def __init__(self, cfg):
        self._cfg = cfg

    def prepare_datasets_case_one(self):
        keys_learn = []
        keys_test = []

        df = pd.read_csv(self._cfg["stead_path_csv"])
        df = df.sample(frac = 1)

        df_eq = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 50)
            & (df.source_magnitude < 3.5)
        ]
        df_an = df[(df.trace_category == "noise")]

        eq_list_learn = df_eq["trace_name"].to_list()[:100000]
        eq_list_test = df_eq["trace_name"].to_list()[100000:150000]
        an_list = df_an["trace_name"].to_list()[:50000]

        for eq in eq_list_learn:
            keys_learn.append(eq)
        
        for eq in eq_list_test:
            keys_test.append(eq)

        for an in an_list:
            keys_test.append(an)
        
        random.shuffle(keys_learn)
        random.shuffle(keys_test)

        len_keys_learn = len(keys_learn)
        len_keys_test = len(keys_test)

        with h5py.File(self._cfg["stead_learn_100hz"], "a") as f:
            f.create_dataset("keys", shape=(len_keys_learn,), dtype="S50")
            f.create_dataset("data", shape=(len_keys_learn, 78, 78, 3), dtype="complex64")
            f.create_dataset("labels", shape=(len_keys_learn,), dtype="i4")

        for event_idx, key in enumerate(keys_learn):
            if event_idx % 100 == 0:
                print(f"{event_idx}/{len_keys_learn}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_learn_100hz"], "a") as f:
                f["keys"][event_idx] = bytes(
                    key, encoding="utf-8"
                )
                f["labels"][event_idx] = 1 if do.trace_name.lower().endswith("ev") else 0
                c = do.get_components()
                _, _, z0 = stft(c[0], window="hanning", fs=100, nperseg=155)
                _, _, z1 = stft(c[1], window="hanning", fs=100, nperseg=155)
                _, _, z2 = stft(c[2], window="hanning", fs=100, nperseg=155)
                arr = np.array([z0, z1, z2])
                arr_t = arr.transpose(1, 2, 0)
                f["data"][event_idx] = arr_t

        with h5py.File(self._cfg["stead_test_100hz"], "a") as f:
            f.create_dataset("keys", shape=(len_keys_test,), dtype="S50")
            f.create_dataset("data", shape=(len_keys_test, 78, 78, 3), dtype="complex64")
            f.create_dataset("labels", shape=(len_keys_test,), dtype="i4")

        for event_idx, key in enumerate(keys_test):
            if event_idx % 100 == 0:
                print(f"{event_idx}/{len_keys_test}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_test_100hz"], "a") as f:
                f["keys"][event_idx] = bytes(
                    key, encoding="utf-8"
                )
                f["labels"][event_idx] = 1 if do.trace_name.lower().endswith("ev") else 0
                c = do.get_components()
                _, _, z0 = stft(c[0], window="hanning", fs=100, nperseg=155)
                _, _, z1 = stft(c[1], window="hanning", fs=100, nperseg=155)
                _, _, z2 = stft(c[2], window="hanning", fs=100, nperseg=155)
                arr = np.array([z0, z1, z2])
                arr_t = arr.transpose(1, 2, 0)
                f["data"][event_idx] = arr_t

    def prepare_datasets_case_two(self):
        keys_learn = []
        keys_test = []

        df = pd.read_csv(self._cfg["stead_path_csv"])
        df = df.sample(frac = 1)

        df_eq = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 50)
            & (df.source_magnitude < 3.5)
        ]
        df_an = df[(df.trace_category == "noise")]

        eq_list_learn = df_eq["trace_name"].to_list()[:100000]
        eq_list_test = df_eq["trace_name"].to_list()[100000:150000]
        an_list = df_an["trace_name"].to_list()[:50000]

        for eq in eq_list_learn:
            keys_learn.append(eq)
        
        for eq in eq_list_test:
            keys_test.append(eq)

        for an in an_list:
            keys_test.append(an)

        random.shuffle(keys_learn)
        random.shuffle(keys_test)

        len_keys_learn = len(keys_learn)
        len_keys_test = len(keys_test)

        with h5py.File(self._cfg["stead_learn_66hz"], "a") as f:
            f.create_dataset("keys", shape=(len_keys_learn,), dtype="S50")
            f.create_dataset("data", shape=(len_keys_learn, 64, 64, 3), dtype="complex64")
            f.create_dataset("labels", shape=(len_keys_learn,), dtype="i4")

        for event_idx, key in enumerate(keys_learn):
            if event_idx % 100 == 0:
                print(f"{event_idx}/{len_keys_learn}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_learn_66hz"], "a") as f:
                f["keys"][event_idx] = bytes(
                    key, encoding="utf-8"
                )
                f["labels"][event_idx] = 1 if do.trace_name.lower().endswith("ev") else 0
                c = do.get_components()
                _, _, z0 = stft(resample(c[0], 4000), window="hanning", fs=66, nperseg=127, return_onesided=True)
                _, _, z1 = stft(resample(c[1], 4000), window="hanning", fs=66, nperseg=127, return_onesided=True)
                _, _, z2 = stft(resample(c[2], 4000), window="hanning", fs=66, nperseg=127, return_onesided=True)
                arr = np.array([z0, z1, z2])
                arr_t = arr.transpose(1, 2, 0)
                f["data"][event_idx] = arr_t
        
        with h5py.File(self._cfg["stead_test_66hz"], "a") as f:
            f.create_dataset("keys", shape=(len_keys_test,), dtype="S50")
            f.create_dataset("data", shape=(len_keys_test, 64, 64, 3), dtype="complex64")
            f.create_dataset("labels", shape=(len_keys_test,), dtype="i4")

        for event_idx, key in enumerate(keys_test):
            if event_idx % 100 == 0:
                print(f"{event_idx}/{len_keys_test}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_test_66hz"], "a") as f:
                f["keys"][event_idx] = bytes(
                    key, encoding="utf-8"
                )
                f["labels"][event_idx] = 1 if do.trace_name.lower().endswith("ev") else 0
                c = do.get_components()
                _, _, z0 = stft(resample(c[0], 4000), window="hanning", fs=66, nperseg=127)
                _, _, z1 = stft(resample(c[1], 4000), window="hanning", fs=66, nperseg=127)
                _, _, z2 = stft(resample(c[2], 4000), window="hanning", fs=66, nperseg=127)
                arr = np.array([z0, z1, z2])
                arr_t = arr.transpose(1, 2, 0)
                f["data"][event_idx] = arr_t

    
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
        df = df.sample(frac = 1)
        df = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 50)
            & (df.source_magnitude < 3.5)
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

    def prepare_mixed_data(self):
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
        with h5py.File(self._cfg["stead_path_db_processed_mixed"], "a") as f:
            f.create_dataset("keys", shape=(len_keys,), dtype="S50")
            f.create_dataset("data", shape=(len_keys, 3, 6000), dtype="float32")
            f.create_dataset("labels", shape=(len_keys,), dtype="i4")

        for idx, key in enumerate(keys):
            if idx % 100 == 0:
                print(f"{idx}/{len_keys}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_path_db_processed_mixed"], "a") as f:
                f["keys"][idx] = bytes(key, encoding="utf-8")
                f["data"][idx] = do.get_components()
                f["labels"][idx] = 1 if do.trace_name.lower().endswith("ev") else 0

    def prepare_gan_data(self):
        df = pd.read_csv(self._cfg["stead_path_csv"])

        df_eq = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 75)
            & (df.source_magnitude > 1.5)
        ]
        eq_list = df_eq["trace_name"].to_list()[:100000]

        random.shuffle(eq_list)
        len_keys = len(eq_list)

        # Init the HDF5 file
        with h5py.File(self._cfg["stead_path_db_processed_gan"], "a") as f:
            f.create_dataset("keys", shape=(len_keys,), dtype="S50")
            f.create_dataset("data", shape=(len_keys, 3, 6000), dtype="float32")

        for idx, key in enumerate(eq_list):
            if idx % 100 == 0:
                print(f"{idx}/{len_keys}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_path_db_processed_gan"], "a") as f:
                f["keys"][idx] = bytes(key, encoding="utf-8")
                f["data"][idx] = do.get_components()

    def prepare_stft_data(self):
        df = pd.read_csv(self._cfg["stead_path_csv"])
        df = df.sample(frac = 1)

        df_eq = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 50)
            & (df.source_magnitude < 3.5)
        ]
        eq_list = df_eq["trace_name"].to_list()[:100000]

        len_events = len(eq_list)
        # We have 3 streams per event
        len_streams = len_events * 3

        # Init the HDF5 file
        with h5py.File(self._cfg["stead_path_db_processed_stft_microseism"], "a") as f:
            f.create_dataset("keys", shape=(len_streams,), dtype="S50")
            f.create_dataset("components", shape=(len_streams,), dtype="S1")
            f.create_dataset("data", shape=(len_streams, 78, 78), dtype="float32")

        for event_idx, key in enumerate(eq_list):
            if event_idx % 100 == 0:
                print(f"{event_idx}/{len_events}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_path_db_processed_stft_microseism"], "a") as f:
                components = do.get_components()
                for component_idx, c in enumerate(components):
                    _, _, Zxx = stft(c, window="hanning", fs=100, nperseg=155)
                    f["keys"][COMP_QTY * event_idx + component_idx] = bytes(
                        key, encoding="utf-8"
                    )
                    f["components"][COMP_QTY * event_idx + component_idx] = bytes(
                        COMP_MAP[component_idx], encoding="utf-8"
                    )
                    f["data"][COMP_QTY * event_idx + component_idx] = Zxx.astype(
                        np.float32
                    )

    def prepare_stft_data_64(self):
        df = pd.read_csv(self._cfg["stead_path_csv"])
        df = df.sample(frac = 1)

        df_eq = df[
            (df.trace_category == "earthquake_local")
            & (df.source_distance_km <= 50)
            & (df.source_magnitude < 3.5)
        ]
        eq_list = df_eq["trace_name"].to_list()[:100000]

        len_events = len(eq_list)
        # We have 3 streams per event
        len_streams = len_events * 3

        # Init the HDF5 file
        with h5py.File(self._cfg["stead_path_db_processed_stft_64"], "a") as f:
            f.create_dataset("keys", shape=(len_streams,), dtype="S50")
            f.create_dataset("components", shape=(len_streams,), dtype="S1")
            f.create_dataset("data", shape=(len_streams, 64, 64), dtype="float32")

        for event_idx, key in enumerate(eq_list):
            if event_idx % 100 == 0:
                print(f"{event_idx}/{len_events}")
            do = self.get_data_by_evi(key)
            with h5py.File(self._cfg["stead_path_db_processed_stft_64"], "a") as f:
                components = do.get_components()
                for component_idx, c in enumerate(components):
                    c_downsampled = resample(c, 4000)
                    _, _, Zxx = stft(c_downsampled, window="hanning", fs=66, nperseg=127)
                    f["keys"][COMP_QTY * event_idx + component_idx] = bytes(
                        key, encoding="utf-8"
                    )
                    f["components"][COMP_QTY * event_idx + component_idx] = bytes(
                        COMP_MAP[component_idx], encoding="utf-8"
                    )
                    f["data"][COMP_QTY * event_idx + component_idx] = Zxx.astype(
                        np.float32
                    )

    def to_dataobject(self, obj) -> SteadDataObject:
        do = SteadDataObject()
        setattr(do, "data", np.array(obj))

        for a in obj.attrs:
            setattr(do, a, obj.attrs[a])

        return do
