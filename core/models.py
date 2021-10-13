from datetime import datetime, timedelta


class DataObject:
    """Placeholder class for data object.
    """
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
        return dt.strftime('%Y-%m-%d %H:%M:%S')
