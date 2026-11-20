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

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("enforce_pinned_compiler.py")


class TestFlagtreeCompilerPin(unittest.TestCase):
    def check_pin(
        self, *, install_package, backend="hygon-dtk2604", version="0.7.0+hcu3.6"
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baked, shadow = root / "flagtree", root / "shadow"
            venv = root / "venv"
            metadata = baked / f"flagtree-{version}.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: flagtree\nVersion: {version}\n",
                encoding="utf-8",
            )
            shadow_pkg = shadow / "triton" / "__init__.py"
            shadow_pkg.parent.mkdir(parents=True)
            shadow_pkg.touch()
            baked_pkg = baked / "triton" / "__init__.py"
            if install_package:
                baked_pkg.parent.mkdir()
                baked_pkg.touch()
            venv_pkg = venv / "triton" / "__init__.py"
            venv_pkg.parent.mkdir(parents=True)
            venv_pkg.touch()
            env = dict(os.environ, VIRTUAL_ENV=str(venv))
            env["PYTHONPATH"] = os.pathsep.join(map(str, (shadow, baked, venv)))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), backend],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertFalse(shadow_pkg.exists(), result.stderr)
            self.assertEqual(baked_pkg.exists(), install_package, result.stderr)
            self.assertTrue(venv_pkg.exists(), result.stderr)
            return result

    def test_preserves_baked_pinned_namespace(self):
        result = self.check_pin(install_package=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_metadata_without_its_namespace(self):
        result = self.check_pin(install_package=False)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("triton resolves to", result.stderr)

    def test_ascend_can_use_separate_namespace(self):
        result = self.check_pin(
            backend="ascend-cann900", version="0.7.1a1+ascend3.5", install_package=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
