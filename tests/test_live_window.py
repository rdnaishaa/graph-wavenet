import unittest

import numpy as np
import pandas as pd

from scripts.live.build_live_latest_12h import prepare_window


class LiveWindowTests(unittest.TestCase):
    def setUp(self):
        self.stations = [f"s{i}" for i in range(64)]
        self.hours = pd.date_range("2026-10-30", periods=12, freq="h", tz="UTC")
        self.now = self.hours[-1] + pd.Timedelta(minutes=30)
        self.df = pd.DataFrame(
            [(h, s, i) for h in self.hours for i, s in enumerate(self.stations)],
            columns=["obs_time_utc", "station_id", "current_speed"],
        )

    def test_training_order_preserved(self):
        _, values = prepare_window(self.df.sample(frac=1), self.stations, self.now)
        np.testing.assert_array_equal(values[0], np.arange(64))
        self.assertEqual(values.shape, (12, 64))

    def test_missing_or_wrong_station_rejected(self):
        with self.assertRaises(RuntimeError):
            prepare_window(self.df.iloc[1:], self.stations, self.now)
        wrong = self.df.replace({"station_id": {"s0": "unknown"}})
        with self.assertRaises(RuntimeError):
            prepare_window(wrong, self.stations, self.now)

    def test_stale_and_invalid_values_rejected(self):
        with self.assertRaises(RuntimeError):
            prepare_window(self.df, self.stations, self.now + pd.Timedelta(hours=3))
        self.df.loc[0, "current_speed"] = -1
        with self.assertRaises(RuntimeError):
            prepare_window(self.df, self.stations, self.now)


if __name__ == "__main__":
    unittest.main()
