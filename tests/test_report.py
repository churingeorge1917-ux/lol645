"""Check that the report figures render without the full source dataset."""

import json
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa

from scope_cache import ROOT
from scope_cache.geometry import MapGeometry
from scope_cache.report import draw_availability, draw_map, draw_smoke


class ReportRenderingTests(unittest.TestCase):
    def setUp(self):
        plt.switch_backend("Agg")

    def tearDown(self):
        plt.close("all")

    def test_availability_figure_renders_from_saved_metadata(self):
        source = json.loads((ROOT / "data/events.metadata.json").read_text())
        figure = draw_availability(source)
        figure.canvas.draw()

    def test_map_figure_renders_with_real_map_assets(self):
        figure = draw_map()
        figure.canvas.draw()

    def test_smoke_figure_renders_without_full_parquet_tables(self):
        geometry = MapGeometry()
        endpoint = (geometry.a[0] + geometry.b[0]) / 2 - [0, 0, 72]
        positions = endpoint + np.array([[0, 0, 0], [16, 0, 0], [0, 16, 0]])
        # Synthetic plotting inputs; not used for model fitting or reported metrics.
        scores = pa.table(
            {
                "potential_smoke_coverage": [0.2, 0.5, 0.7],
                "smoke_coverage_radius96": [0.1, 0.25, 0.4],
                "smoke_coverage_radius192": [0.4, 0.7, 0.9],
                "candidate_static_sightlines": [100, 100, 100],
            }
        )
        endpoints = pa.table(
            {"end_" + axis: positions[:, i] for i, axis in enumerate("xyz")}
        )
        with patch("scope_cache.report.pq.read_table", side_effect=[scores, endpoints]):
            figure = draw_smoke()
        figure.canvas.draw()


if __name__ == "__main__":
    unittest.main()
