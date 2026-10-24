### технический файл, чтобы пофиксить ошибику при загрузке данных на гитхаб. не стоит рассматривать как что-то важное для проекта
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

BEFORE = {
    ".gitignore": "78715016cacf748f9e004416e414736f83d6de55d41a45748fc021153905897a",
    "ml_algorithm.py": "ffbe26694ffd5a26ffe2702456ef1a8a603ed5898c1d8c2d01b8008fded306cc",
    "test_poc.py": "4755bc1de3a684a33748701405f57dd2a9330f98449cbf63ac915fb874c9d878",
    "scope_cache/report.py": "bbbcd92229afa60745b3d00de40051441af47f2496a0e41180d39ab39751bb16"
}

AFTER = {
    ".gitignore": "456bd8756e41ffbbf6a4a9908f5fb3e24cb676d78ccbc1d3231fc2cb871cc2b6",
    "ml_algorithm.py": "89d28989457d713ef1096feee597f0e1c6c34de54ad88347d52bfeaad4589c8d",
    "test_poc.py": "c8eae86e8e31987e270d2013d5b88f93a11bdf8dee8b8b948edb3b7134ae6fec",
    "scope_cache/report.py": "3315e497b1fcc49c376730167bb48e9660f13ce8d5c4accd38501dafc5754cb4",
    "map_assets/manifest.json": "d98f7415de760d385d3b8e57e3fc1d495af54f3829784853d224fbb0dd96f638",
    "tests/test_report.py": "9d909ab0ef71fb1860cd3c97e220a9fab75e044cbc2b2356a044d77eb4109b98"
}


def digest(path):
    if not path.is_file():
        return None
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main():
    root = Path.cwd().resolve()
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise RuntimeError("Run from the project root containing scope.py")
    if git(root, "branch", "--show-current") != "developer":
        raise RuntimeError("Switch to developer first: git switch developer")
    exclusions = []
    for relative, expected in AFTER.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError(f"Unexpected symlink: {relative}")
        actual = digest(path)
        if actual == expected:
            exclusions.append(relative)
        elif actual != BEFORE.get(relative):
            raise RuntimeError(
                f"File differs from the checked repository: {relative}. "
                "No files have been changed."
            )
    if len(exclusions) < len(AFTER):
        with tempfile.NamedTemporaryFile(suffix=".patch", delete=False) as handle:
            handle.write(PATCH.encode("utf-8"))
            patch_path = Path(handle.name)
        try:
            options = ["--index", "--ignore-space-change"]
            options.extend("--exclude=" + name for name in exclusions)
            git(root, "apply", "--check", *options, str(patch_path))
            git(root, "apply", *options, str(patch_path))
        finally:
            patch_path.unlink(missing_ok=True)
    for relative, expected in AFTER.items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"Unexpected result: {relative}")
    git(root, "add", "--", *BEFORE)
    git(root, "add", "-f", "--", *(name for name in AFTER if name not in BEFORE))
    print("CI fixes applied and staged. Model weights and data were not changed.")
    print('Next: git commit -m "Fix CI checks and report rendering"')
    print("Then: git push origin developer")


PATCH = r'''diff --git a/.gitignore b/.gitignore
index a7674478a55700e0f4ee4d0e3a37779fba27a53c..9a541d697f068fcd015e5e83ae62787019bfa9f9 100644
--- a/.gitignore
+++ b/.gitignore
@@ -10,7 +10,7 @@
 *.tmp
 *.list
 *.zip
-MANIFEST.json
+/MANIFEST.json
 data/*.parquet
 demo_context/
 figures/
diff --git a/ml_algorithm.py b/ml_algorithm.py
index b1401c02e446221f0cc00924034155a9cd80d8fe..ec44c72ff195492d504f201c3dfe2dd0ab3cdc28 100644
--- a/ml_algorithm.py
+++ b/ml_algorithm.py
@@ -162,7 +162,7 @@
 def refresh_player_matches(source: Path, target: Path) -> dict:
     """Пересчитать полный профиль из исходника, не меняя модель и её выборку."""
     with database(target) as connection:
-        provenance = read_document(connection, "source")
+        read_document(connection, "source")
         extraction = read_document(connection, "extraction")
     with source.open("rb") as stream:
         source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
diff --git a/test_poc.py b/test_poc.py
index afd3df8fc01cab065798a2e66f44cd33490fc17d..976122e4eac9e8be1835b2147f9354961b47a04e 100644
--- a/test_poc.py
+++ b/test_poc.py
@@ -8,8 +8,9 @@
 from pathlib import Path
 from unittest.mock import patch
 
+import ml_algorithm as model
+
 ROOT = Path(__file__).resolve().parent
-import ml_algorithm as model
 
 
 def sample_match():
diff --git a/scope_cache/report.py b/scope_cache/report.py
index c72094122ea6ade17d22d8e6a5ab44f414fe2349..c5ce0c31e33ce553d8098532cc07f74932016978 100644
--- a/scope_cache/report.py
+++ b/scope_cache/report.py
@@ -71,11 +71,6 @@
     ax.grid(axis="x", alpha=0.15)
     ax.set_axisbelow(True)
     fig.subplots_adjust(left=0.20, right=0.97, top=0.84, bottom=0.27)
-    fig.text(
-        0.06,
-        0.04,
-        fontsize=10,
-    )
     return fig
 
 
@@ -141,16 +136,6 @@
         y=0.98,
     )
     fig.subplots_adjust(left=0.07, right=0.96, top=0.85, bottom=0.16, wspace=0.12)
-    fig.text(
-        0.06,
-        0.08,
-        fontsize=10,
-    )
-    fig.text(
-        0.06,
-        0.035,
-        fontsize=10,
-    )
     return fig
 
 
@@ -256,11 +241,6 @@
     )
     fig.text(
         0.06,
-        0.13,
-        fontsize=10,
-    )
-    fig.text(
-        0.06,
         0.08,
         f"729 947 смоков; {int((~finite).sum()):,} без подходящих линий. Радиусы — сценарии чувствительности".replace(
             ",", " "
@@ -327,7 +307,7 @@
             xlim=(0, 1),
             ylim=(0, 1),
             xlabel="Доля перекрытых линий",
-            ylabel=f"Доля перекрытых линий",
+            ylabel="Доля перекрытых линий",
             title=f"Шаг {step} · Spearman {setting['spearman_rank_correlation']:.3f} · n={valid.sum()}",
         )
         ax.set_aspect("equal")
diff --git a/map_assets/manifest.json b/map_assets/manifest.json
new file mode 100644
index 0000000000000000000000000000000000000000..b00987931317763240aa454c8eed6e738439fdcf
--- /dev/null
+++ b/map_assets/manifest.json
@@ -0,0 +1,53 @@
+{
+  "checked_at": "2026-10-23",
+  "client_version": "2000934",
+  "patch_version": "1.41.9.1",
+  "buildid": "26087496",
+  "source_revision": "11150591",
+  "release_url": "https://github.com/pnxenopoulos/awpy-data/releases/tag/2000934",
+  "release_generated_at": "2026-10-21T06:08:15.872467+00:00",
+  "coordinate_system": "Hammer XYZ, Z-up",
+  "dataset_version_match": "unverified: no source game version in input",
+  "geometry": "static, material-filtered and decimated collision mesh; no moving doors, smoke physics or penetration",
+  "game_version_evidence": "https://github.com/SteamTracking/GameTracking-CS2/blob/master/game/csgo/steam.inf",
+  "files": {
+    "de_cache.mesh": {
+      "sha256": "76193544556d261eb1ef57730cd19de5ab648713d0451ec689fbefe9271aa702",
+      "bytes": 6237892
+    },
+    "de_cache.nav": {
+      "sha256": "a3dd9d3b9b0be9fa319e7f0ab035238ea7987b9773d21e49d25438e7e8a8b0b2",
+      "bytes": 433020
+    },
+    "de_cache.png": {
+      "sha256": "94c058b3deaa5cc24be81006322da3af1ef0bbded116160a3fcdea6d87227967",
+      "bytes": 93413
+    },
+    "sightlines.npz": {
+      "sha256": "722cb407d3d3618f7e02923da604372cc9715cf69201d4aeedae562f85fccac5",
+      "bytes": 56348
+    },
+    "radar_transform.json": {
+      "sha256": "3354ea8a320b898666693b03e6b31caf728d9bc9057a40acd8011b3579db8e16",
+      "bytes": 92
+    },
+    "geometry_report.json": {
+      "sha256": "35c019c7eb8f54634677cd6e67c6cbb1c0bde2dab61c88f4e1636a791266c9fd",
+      "bytes": 1043
+    }
+  },
+  "release_artifacts": {
+    "geometry.zip": {
+      "bytes": 56674290,
+      "sha256": "edfae7235b87e6ca4cb49c453777a40991f5ba1ce438b8d9f14128bd0a8c56b3"
+    },
+    "images.zip": {
+      "bytes": 2876805,
+      "sha256": "3824c0330e628666fe63d6165335aa34809464f0c113dbc758bce3ca4ec4adbb"
+    },
+    "navs.zip": {
+      "bytes": 8432990,
+      "sha256": "90a353d8d0f4af9368898e530bdc9b0620801d4beafc3126999d27882569b1fa"
+    }
+  }
+}
diff --git a/tests/test_report.py b/tests/test_report.py
new file mode 100644
index 0000000000000000000000000000000000000000..d1eb851b6724053aaa9458637959b1e4c43f0b63
--- /dev/null
+++ b/tests/test_report.py
@@ -0,0 +1,54 @@
+"""Check that the report figures render without the full source dataset."""
+
+import json
+import unittest
+from unittest.mock import patch
+
+import matplotlib.pyplot as plt
+import numpy as np
+import pyarrow as pa
+
+from scope_cache import ROOT
+from scope_cache.geometry import MapGeometry
+from scope_cache.report import draw_availability, draw_map, draw_smoke
+
+
+class ReportRenderingTests(unittest.TestCase):
+    def setUp(self):
+        plt.switch_backend("Agg")
+
+    def tearDown(self):
+        plt.close("all")
+
+    def test_availability_figure_renders_from_saved_metadata(self):
+        source = json.loads((ROOT / "data/events.metadata.json").read_text())
+        figure = draw_availability(source)
+        figure.canvas.draw()
+
+    def test_map_figure_renders_with_real_map_assets(self):
+        figure = draw_map()
+        figure.canvas.draw()
+
+    def test_smoke_figure_renders_without_full_parquet_tables(self):
+        geometry = MapGeometry()
+        endpoint = (geometry.a[0] + geometry.b[0]) / 2 - [0, 0, 72]
+        positions = endpoint + np.array([[0, 0, 0], [16, 0, 0], [0, 16, 0]])
+        # Synthetic plotting inputs; not used for model fitting or reported metrics.
+        scores = pa.table(
+            {
+                "potential_smoke_coverage": [0.2, 0.5, 0.7],
+                "smoke_coverage_radius96": [0.1, 0.25, 0.4],
+                "smoke_coverage_radius192": [0.4, 0.7, 0.9],
+                "candidate_static_sightlines": [100, 100, 100],
+            }
+        )
+        endpoints = pa.table(
+            {"end_" + axis: positions[:, i] for i, axis in enumerate("xyz")}
+        )
+        with patch("scope_cache.report.pq.read_table", side_effect=[scores, endpoints]):
+            figure = draw_smoke()
+        figure.canvas.draw()
+
+
+if __name__ == "__main__":
+    unittest.main()
'''


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
