# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free tests of model source defaults; no GPU/model imports."""

import ast
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/flag_gems/flagtune/inference/cost_model.py"
MANIFEST = SOURCE.with_name("manifest.json")


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        # Execute the actual initializer without importing optional GPU packages.
        tree = ast.parse(SOURCE.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_configure_flagtune_model_source"
        )
        self.namespace = {
            "os": os,
            "_MODEL_SOURCE_CONFIGURED": False,
            "_DEFAULT_FLAGTUNE_LOCAL_MANIFEST": MANIFEST,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
            self.namespace,
        )
        self.configure = self.namespace["_configure_flagtune_model_source"]

    def test_defaults_and_once_only(self):
        with patch.dict(os.environ, {}, clear=True):
            self.configure()
            self.assertEqual(os.environ["FLAGTUNE_LOCAL_MANIFEST"], str(MANIFEST))
            self.assertNotIn("FLAGTUNE_MANIFEST_URL", os.environ)
            self.assertEqual(os.environ["FLAGTUNE_MODEL_DOWNLOAD_LATEST"], "1")
            del os.environ["FLAGTUNE_LOCAL_MANIFEST"]
            self.configure()
            self.assertNotIn("FLAGTUNE_LOCAL_MANIFEST", os.environ)

    def test_explicit_sources_including_empty_are_preserved(self):
        for settings in (
            {"FLAGTUNE_LOCAL_MANIFEST": "/custom/manifest.json"},
            {"FLAGTUNE_MANIFEST_URL": "https://example.com/manifest.tar.gz"},
            {"FLAGTUNE_LOCAL_MANIFEST": ""},
            {"FLAGTUNE_MANIFEST_URL": ""},
            {"FLAGTUNE_LOCAL_MANIFEST": "/custom.json", "FLAGTUNE_MANIFEST_URL": ""},
        ):
            with (
                self.subTest(settings=settings),
                patch.dict(os.environ, settings, clear=True),
            ):
                self.namespace["_MODEL_SOURCE_CONFIGURED"] = False
                self.configure()
                for key, value in settings.items():
                    self.assertEqual(os.environ[key], value)
                if "FLAGTUNE_LOCAL_MANIFEST" not in settings:
                    self.assertNotIn("FLAGTUNE_LOCAL_MANIFEST", os.environ)

    def test_version_preferences_preserved(self):
        with patch.dict(
            os.environ,
            {"FLAGTUNE_MODEL_DOWNLOAD_LATEST": "0", "FLAGTUNE_MODEL_VERSION": "1.0.0"},
            clear=True,
        ):
            self.configure()
            self.assertEqual(os.environ["FLAGTUNE_MODEL_DOWNLOAD_LATEST"], "0")
            self.assertEqual(os.environ["FLAGTUNE_MODEL_VERSION"], "1.0.0")

    def test_manifest_schema_and_wheel_data(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["packages"]), 5)
        for platform, package in manifest["packages"].items():
            for version, item in package["versions"].items():
                self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(item["url"].startswith("https://"))
                self.assertTrue(item["url"].endswith(f"{platform}_v{version}.tar.gz"))
        self.assertIn(
            '"flag_gems.flagtune.inference" = ["manifest.json"]',
            (ROOT / "pyproject.toml").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
