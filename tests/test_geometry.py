import unittest
from types import SimpleNamespace
from scope_cache.geometry import (
    MapGeometry,
    read_mesh,
    segment_distances,
    smoke_chords,
    verify_assets,
)


class GeometryTests(unittest.TestCase):
    def test_ellipsoid_chords_use_finite_segments(self):
        self.assertAlmostEqual(float(smoke_chords([-2, 0, 0], [2, 0, 0], [0, 0, 0], 1, 1)), 2)
        self.assertEqual(float(smoke_chords([-2, 0, 0], [-1.5, 0, 0], [0, 0, 0], 1, 1)), 0)
        self.assertEqual(float(smoke_chords([-2, 1, 0], [2, 1, 0], [0, 0, 0], 1, 1)), 0)
        self.assertAlmostEqual(float(smoke_chords([0, 0, 0], [2, 0, 0], [0, 0, 0], 1, 1)), 1)
        self.assertEqual(float(smoke_chords([0, 0, 0], [0, 0, 0], [0, 0, 0], 1, 1)), 0)
        with self.assertRaises(ValueError):
            smoke_chords([0, 0, 0], [1, 0, 0], [0, 0, 0], 0, 1)

    def test_height_is_not_projected_away(self):
        self.assertEqual(float(smoke_chords([-2, 0, 2], [2, 0, 2], [0, 0, 0], 1, 1)), 0)
        self.assertAlmostEqual(float(segment_distances([0, 0, 0], [1, 0, 0], [2, 0, 0])), 1)

    def test_real_assets_and_wall_aware_visibility(self):
        verify_assets()
        vertices, triangles = read_mesh()
        self.assertEqual(len(triangles), 338375)
        geometry = MapGeometry()
        self.assertTrue(geometry.visible(geometry.a[0], geometry.b[0]))
        self.assertTrue(
            any(not geometry.visible(point, point - [0, 0, 200]) for point in geometry.a[:100])
        )
        value = geometry.smoke_at((geometry.a[0] + geometry.b[0]) / 2 - [0, 0, 72])
        self.assertIsNone(value["actual_enemy_view_blocked"])
        self.assertLessEqual(value["coverage_radius_96"], value["potential_coverage"])
        self.assertLessEqual(value["potential_coverage"], value["coverage_radius_192"])

    def test_snapshot_requires_synchronous_telemetry(self):
        geometry = MapGeometry.__new__(MapGeometry)
        geometry.visibility = SimpleNamespace(is_visible=lambda a, b: True)
        players = [
            {
                "side": 2,
                "eye": [-200, 0, 72],
                "gaze": [1, 0, 0],
                "fov_degrees": 90,
                "tick": 100,
            },
            {
                "side": 3,
                "eye": [200, 0, 72],
                "gaze": [-1, 0, 0],
                "fov_degrees": 90,
                "tick": 100,
            },
        ]
        result = geometry.snapshot_blocking(
            [0, 0, 0], players, tick=100, smoke_begin=90, smoke_end=110
        )
        self.assertEqual(result["blocked_pairs"], 2)
        players[1]["tick"] = 99
        with self.assertRaises(ValueError):
            geometry.snapshot_blocking([0, 0, 0], players, tick=100, smoke_begin=90, smoke_end=110)
        self.assertEqual(
            geometry.snapshot_blocking([0, 0, 0], [], tick=120, smoke_begin=90, smoke_end=110)[
                "status"
            ],
            "inactive_smoke",
        )
