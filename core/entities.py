import numpy as np
from datetime import datetime, timedelta
from scipy.signal import spectrogram


class SteadDataObject:
    def __init__(self):
        pass

    def get_component(self, component: str) -> list:
        # There are 3 channels: first row: E channel, second row: N channel, third row: Z channel
        rotated = np.rot90(self.data, k=1, axes=(0,1))
        if component.lower() == "e":
            return rotated[2]
        elif component.lower() == "n":
            return rotated[1]
        else:
            return rotated[0]

    def get_components(self):
        rotated = np.rot90(self.data, k=1, axes=(0,1))
        return rotated

    def get_timespan(self):
        t = self.get_ts_datetime()
        time_points = []
        for _ in range(6000):
            time_points.append(t)
            t += timedelta(seconds=0.01)
        return time_points

    def get_ts_datetime(self):
        dt = datetime.strptime(self.trace_start_time, "%Y-%m-%d %H:%M:%S.%f")
        return dt

    def get_ts_iso8601(self):
        dt = self.get_ts_datetime()
        return dt.isoformat()

    def get_ts_short(self):
        dt = self.get_ts_datetime()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
class LenDataObject:
    """Placeholder class for data object."""

    def __init__(self):
        self.id = None
        self.type = None
        self.az = None
        self.baz = None
        self.dist = None
        self.endtime = None
        self.evdp = None
        self.evla = None
        self.evlo = None
        self.mag = None
        self.net = None
        self.otime = None
        self.sta = None
        self.starttime = None
        self.stel = None
        # List of 3 seismograms of 27​s sampled at 20 ​Hz.
        self.data = []

    def get_timespan(self):
        t = self.get_ts_datetime()
        time_points = []
        for _ in range(540):
            time_points.append(t)
            t += timedelta(seconds=0.05)
        return time_points

    def get_ts_datetime(self):
        dt = datetime.utcfromtimestamp(self.starttime)
        return dt

    def get_ts_iso8601(self):
        dt = self.get_ts_datetime()
        return dt.isoformat()

    def get_ts_short(self):
        dt = self.get_ts_datetime()
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def get_flattened_data(self):
        return np.array(self.data).reshape(1620)

    def get_spectrograms(self):
        spectrograms = []
        for d in self.data:
            spectrograms.append(
                [spectrogram(d, fs=20)]
            )
        return spectrograms
