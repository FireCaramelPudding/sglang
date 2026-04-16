"""
Minimal end-to-end smoke test for native KV export/graft HTTP flow.

This test validates the smallest useful closed loop:
1. Request A generates tokens and exports an answer-only KV handle.
2. Request B grafts that handle into a new request and re-exports the merged prefix.
3. The debug and release endpoints work for both exported handles.

Run with:
    PYTHONPATH=/ssd/home/xiaoliangyang/sglang/python \
    SGLANG_KV_GRAFT_TEST_MODEL=/ssd/home/xiaoliangyang/models/Qwen/Qwen2.5-3B-Instruct \
    python3 test/manual/entrypoints/http_server/test_kv_graft_smoke.py
"""

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    auto_config_device,
    find_available_port,
)

LOCAL_QWEN_MODEL = "/ssd/home/xiaoliangyang/models/Qwen/Qwen2.5-3B-Instruct"
TEST_MODEL = os.environ.get(
    "SGLANG_KV_GRAFT_TEST_MODEL",
    LOCAL_QWEN_MODEL
    if os.path.isdir(LOCAL_QWEN_MODEL)
    else DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
)


class TestKVGraftSmoke(unittest.TestCase):
    model = TEST_MODEL
    request_timeout = 120
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    first_input_ids = [100, 101, 102, 103, 104]
    second_input_ids = [201, 202]

    @classmethod
    def setUpClass(cls):
        port = find_available_port(DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.health_url = f"{cls.base_url}/health_generate"
        cls._session = requests.Session()
        device = os.environ.get("SGLANG_KV_GRAFT_TEST_DEVICE") or auto_config_device()

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{PYTHON_ROOT}:{existing_pythonpath}"
            if existing_pythonpath
            else str(PYTHON_ROOT)
        )

        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-cuda-graph",
            "--device",
            device,
        ]

        print(f"command={' '.join(command)}")
        cls.process = subprocess.Popen(command, env=env)
        cls._wait_for_server_ready()

    @classmethod
    def tearDownClass(cls):
        try:
            cls._session.close()
        except Exception:
            pass
        kill_process_tree(cls.process.pid)

    @classmethod
    def _wait_for_server_ready(cls):
        start = time.perf_counter()
        while time.perf_counter() - start < cls.server_timeout:
            return_code = cls.process.poll()
            if return_code is not None:
                raise RuntimeError(f"Server exited early with code {return_code}")

            try:
                response = cls._session.get(cls.health_url, timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            time.sleep(5)

        raise TimeoutError(
            f"Server did not become healthy within {cls.server_timeout} seconds"
        )

    def setUp(self):
        response = self._session.post(f"{self.base_url}/flush_cache", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        self._handles_to_release = []

    def tearDown(self):
        handles = getattr(self, "_handles_to_release", [])
        if not handles:
            return

        try:
            requests.post(
                f"{self.base_url}/kv_handles/release",
                json={"handles": handles},
                timeout=30,
            )
        except requests.RequestException:
            pass

    def _post_generate(self, payload):
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.request_timeout,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _extract_single_kv_export(self, response_json):
        exports = response_json.get("kv_exports") or response_json.get(
            "meta_info", {}
        ).get("kv_exports", [])
        self.assertEqual(len(exports), 1, response_json)
        return exports[0]

    def _get_handle(self, handle, expected_status=200):
        response = requests.get(
            f"{self.base_url}/kv_handles/{handle}",
            timeout=30,
        )
        self.assertEqual(response.status_code, expected_status, response.text)
        return response.json()

    def _release_handles(self, handles):
        response = requests.post(
            f"{self.base_url}/kv_handles/release",
            json={"handles": handles},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["success"], body)
        return body

    def test_native_export_graft_reexport_release_smoke(self):
        first_response = self._post_generate(
            {
                "input_ids": self.first_input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 2,
                    "min_new_tokens": 2,
                    "ignore_eos": True,
                },
                "kv_export": {
                    "token_start": len(self.first_input_ids),
                    "origin_start": 0,
                    "persist": True,
                    "ttl_seconds": 300,
                    "name": "smoke-answer-a",
                },
            }
        )
        first_export = self._extract_single_kv_export(first_response)
        first_handle = first_export["handle"]
        self._handles_to_release.append(first_handle)

        self.assertTrue(first_handle)
        self.assertGreater(first_export["token_count"], 0)
        self.assertEqual(first_export["origin_start"], 0)

        first_handle_debug = self._get_handle(first_handle)
        first_meta = first_handle_debug["handle_meta"]
        self.assertTrue(first_handle_debug["success"])
        self.assertEqual(first_meta["handle"], first_handle)
        self.assertEqual(first_meta["token_count"], first_export["token_count"])
        self.assertEqual(first_meta["origin_start"], first_export["origin_start"])
        self.assertFalse(first_meta["composite"])

        second_response = self._post_generate(
            {
                "input_ids": self.second_input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "min_new_tokens": 1,
                    "ignore_eos": True,
                },
                "kv_graft": {
                    "segments": [
                        {
                            "handle": first_handle,
                            "origin_start": first_export["origin_start"],
                        }
                    ]
                },
                "kv_export": {
                    "origin_start": 0,
                    "token_end": first_export["token_count"] + len(self.second_input_ids),
                    "persist": True,
                    "ttl_seconds": 300,
                    "name": "smoke-merged-prefix-b",
                },
            }
        )
        second_export = self._extract_single_kv_export(second_response)
        second_handle = second_export["handle"]
        self._handles_to_release.append(second_handle)

        expected_merged_token_count = (
            first_export["token_count"] + len(self.second_input_ids)
        )
        self.assertTrue(second_handle)
        self.assertNotEqual(second_handle, first_handle)
        self.assertEqual(second_export["token_count"], expected_merged_token_count)
        self.assertEqual(second_export["origin_start"], 0)

        second_handle_debug = self._get_handle(second_handle)
        second_meta = second_handle_debug["handle_meta"]
        self.assertTrue(second_handle_debug["success"])
        self.assertEqual(second_meta["handle"], second_handle)
        self.assertEqual(second_meta["token_count"], expected_merged_token_count)
        self.assertTrue(second_meta["composite"])

        release_body = self._release_handles([first_handle, second_handle])
        self.assertCountEqual(
            release_body["released_handles"], [first_handle, second_handle]
        )
        self.assertEqual(release_body["missing_handles"], [])
        self._handles_to_release = []

        first_missing = self._get_handle(first_handle, expected_status=404)
        second_missing = self._get_handle(second_handle, expected_status=404)
        self.assertFalse(first_missing["success"])
        self.assertFalse(second_missing["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
