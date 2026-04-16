"""Manual KV tensor equivalence test for native export vs graft vs local HF.

Case A:
    Direct prefill/export on full input_ids.
Case B:
    Export prefix handle, then graft prefix handle + suffix input_ids and export merged KV.
Case C:
    Local HuggingFace prefill on same full input_ids.

Primary goal:
    Compare real KV tensors behind exported handles instead of generated text.

Run with:
    PYTHONPATH=/ssd/home/xiaoliangyang/sglang/python \
    SGLANG_KV_GRAFT_TEST_MODEL=/ssd/home/xiaoliangyang/models/Qwen/Qwen2.5-3B-Instruct \
    python3 test/manual/entrypoints/http_server/test_kv_graft_tensor_equivalence.py

Optional env:
    SGLANG_KV_GRAFT_TEST_DEVICE      Server device. Defaults to auto-configured device.
    SGLANG_KV_GRAFT_TEST_HF_DEVICE   HF device. Defaults to cpu to avoid GPU OOM.
    SGLANG_KV_GRAFT_TEST_DETERMINISTIC
                                      Defaults to 1. Set to 0 to use server default behavior.
    SGLANG_KV_GRAFT_TEST_ATTENTION_BACKEND
                                      Defaults to triton when deterministic mode is on.
    SGLANG_KV_GRAFT_TEST_DISABLE_PIECEWISE_CUDA_GRAPH
                                      Defaults to 0. Set to 1 for extra determinism diagnostics.
"""

import gc
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
HF_DEVICE = os.environ.get("SGLANG_KV_GRAFT_TEST_HF_DEVICE", "cpu")
DETERMINISTIC_MODE = (
    os.environ.get("SGLANG_KV_GRAFT_TEST_DETERMINISTIC", "1").strip().lower()
    not in {"0", "false", "no"}
)
DEFAULT_ATTENTION_BACKEND = "triton" if DETERMINISTIC_MODE else None
ATTENTION_BACKEND = os.environ.get(
    "SGLANG_KV_GRAFT_TEST_ATTENTION_BACKEND", DEFAULT_ATTENTION_BACKEND
)
DISABLE_PIECEWISE_CUDA_GRAPH = (
    os.environ.get("SGLANG_KV_GRAFT_TEST_DISABLE_PIECEWISE_CUDA_GRAPH", "0")
    .strip()
    .lower()
    in {"1", "true", "yes"}
)


class TestKVGraftTensorEquivalence(unittest.TestCase):
    model = TEST_MODEL
    request_timeout = 180
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    server_dtype = "auto"
    full_text = "KV graft tensor equivalence test. Alpha beta gamma delta epsilon zeta."
    direct_vs_graft_atol = 1e-6
    hf_atol = 5e-2

    @classmethod
    def setUpClass(cls):
        port = find_available_port(DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1100)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.health_url = f"{cls.base_url}/health_generate"
        cls._session = requests.Session()
        cls._handles_to_release = []

        server_device = os.environ.get("SGLANG_KV_GRAFT_TEST_DEVICE") or auto_config_device()
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{PYTHON_ROOT}:{existing_pythonpath}"
            if existing_pythonpath
            else str(PYTHON_ROOT)
        )

        command = [
            sys.executable,
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
            server_device,
        ]
        if DETERMINISTIC_MODE:
            command.append("--enable-deterministic-inference")
        if ATTENTION_BACKEND:
            command.extend(["--attention-backend", ATTENTION_BACKEND])
        if DISABLE_PIECEWISE_CUDA_GRAPH:
            command.append("--disable-piecewise-cuda-graph")
        print(
            "server_config="
            f"deterministic={DETERMINISTIC_MODE} "
            f"attention_backend={ATTENTION_BACKEND or 'default'} "
            f"disable_piecewise_cuda_graph={DISABLE_PIECEWISE_CUDA_GRAPH} "
            f"device={server_device}"
        )
        print(f"command={' '.join(command)}")
        cls.process = subprocess.Popen(command, env=env)
        cls._wait_for_server_ready()

        print(f"Loading HF tokenizer from {cls.model}")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model,
            trust_remote_code=True,
            use_fast=True,
        )

        hf_dtype = torch.float32 if HF_DEVICE == "cpu" else torch.bfloat16
        print(f"Loading HF model from {cls.model} on {HF_DEVICE} with dtype={hf_dtype}")
        cls.hf_model = (
            AutoModelForCausalLM.from_pretrained(
                cls.model,
                trust_remote_code=True,
                torch_dtype=hf_dtype,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(HF_DEVICE)
        )

    @classmethod
    def tearDownClass(cls):
        try:
            handles = list(dict.fromkeys(getattr(cls, "_handles_to_release", [])))
            if handles:
                cls._session.post(
                    f"{cls.base_url}/kv_handles/release",
                    json={"handles": handles},
                    timeout=30,
                )
        except Exception:
            pass

        try:
            cls._session.close()
        except Exception:
            pass

        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

        if hasattr(cls, "hf_model"):
            del cls.hf_model
        if hasattr(cls, "tokenizer"):
            del cls.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        self._handles_to_release_local = []

    def tearDown(self):
        handles = list(dict.fromkeys(getattr(self, "_handles_to_release_local", [])))
        if not handles:
            return
        try:
            self._session.post(
                f"{self.base_url}/kv_handles/release",
                json={"handles": handles},
                timeout=30,
            )
        except requests.RequestException:
            pass

    def _track_handle(self, handle: str):
        self._handles_to_release_local.append(handle)
        self.__class__._handles_to_release.append(handle)

    def _post_generate(self, payload: Dict) -> Dict:
        response = self._session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.request_timeout,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _extract_single_kv_export(self, response_json: Dict) -> Dict:
        exports = response_json.get("kv_exports") or response_json.get(
            "meta_info", {}
        ).get("kv_exports", [])
        self.assertEqual(len(exports), 1, response_json)
        return exports[0]

    def _fetch_handle_tensors(self, handle: str) -> Dict:
        layer_ids = ",".join(
            str(i) for i in range(self.hf_model.config.num_hidden_layers)
        )
        response = self._session.get(
            f"{self.base_url}/kv_handles/{handle}/tensors",
            params={"layer_ids": layer_ids},
            timeout=self.request_timeout,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["success"], body)
        self.assertTrue(body["layers"], body)
        return body

    def _server_layer_map(self, handle_tensors: Dict) -> Dict[int, Dict[str, torch.Tensor]]:
        layer_map = {}
        for layer in handle_tensors["layers"]:
            layer_id = int(layer["layer_id"])
            item = {}
            for key, value in layer.items():
                if key == "layer_id":
                    continue
                item[key] = torch.tensor(value, dtype=torch.float32)
            layer_map[layer_id] = item
        return layer_map

    def _hf_past_key_values(self, input_ids: List[int]):
        model_input = torch.tensor([input_ids], dtype=torch.long, device=HF_DEVICE)
        with torch.inference_mode():
            outputs = self.hf_model(model_input, use_cache=True)
        past_key_values = outputs.past_key_values
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        return past_key_values

    def _hf_mha_layer_map(self, input_ids: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
        past_key_values = self._hf_past_key_values(input_ids)
        layer_map = {}
        for layer_id, layer_cache in enumerate(past_key_values):
            if len(layer_cache) < 2:
                raise RuntimeError(f"Unexpected HF past_key_values item at layer {layer_id}")
            k = layer_cache[0].detach().float().cpu()
            v = layer_cache[1].detach().float().cpu()
            if k.dim() != 4 or v.dim() != 4:
                raise RuntimeError(
                    f"Unexpected HF KV shape at layer {layer_id}: k={tuple(k.shape)} v={tuple(v.shape)}"
                )
            layer_map[layer_id] = {
                "k": k[0].permute(1, 0, 2).contiguous(),
                "v": v[0].permute(1, 0, 2).contiguous(),
            }
        return layer_map

    def _compare_tensor_pair(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        atol: float,
    ) -> Dict:
        if tuple(lhs.shape) != tuple(rhs.shape):
            return {
                "allclose": False,
                "max_abs_diff": None,
                "shape_lhs": tuple(lhs.shape),
                "shape_rhs": tuple(rhs.shape),
                "first_bad_index": None,
                "lhs_value": None,
                "rhs_value": None,
            }

        if lhs.numel() == 0:
            return {
                "allclose": True,
                "max_abs_diff": 0.0,
                "shape_lhs": tuple(lhs.shape),
                "shape_rhs": tuple(rhs.shape),
                "first_bad_index": None,
                "lhs_value": None,
                "rhs_value": None,
            }

        diff = (lhs - rhs).abs()
        max_abs_diff = float(diff.max().item())
        allclose = bool(torch.allclose(lhs, rhs, atol=atol, rtol=0))
        first_bad_index = None
        lhs_value = None
        rhs_value = None
        if not allclose:
            bad = (diff > atol).nonzero(as_tuple=False)
            if bad.numel() > 0:
                first_bad_index = tuple(int(x) for x in bad[0].tolist())
                lhs_value = float(lhs[first_bad_index].item())
                rhs_value = float(rhs[first_bad_index].item())
        return {
            "allclose": allclose,
            "max_abs_diff": max_abs_diff,
            "shape_lhs": tuple(lhs.shape),
            "shape_rhs": tuple(rhs.shape),
            "first_bad_index": first_bad_index,
            "lhs_value": lhs_value,
            "rhs_value": rhs_value,
        }

    def _assert_layerwise_allclose(
        self,
        *,
        lhs_name: str,
        rhs_name: str,
        lhs_layers: Dict[int, Dict[str, torch.Tensor]],
        rhs_layers: Dict[int, Dict[str, torch.Tensor]],
        tensor_keys: Tuple[str, ...],
        atol: float,
    ):
        failure_messages = []
        for layer_id in sorted(lhs_layers.keys()):
            self.assertIn(layer_id, rhs_layers, f"Missing layer {layer_id} in {rhs_name}")
            for tensor_key in tensor_keys:
                self.assertIn(
                    tensor_key,
                    lhs_layers[layer_id],
                    f"Missing {tensor_key} in {lhs_name} layer {layer_id}",
                )
                self.assertIn(
                    tensor_key,
                    rhs_layers[layer_id],
                    f"Missing {tensor_key} in {rhs_name} layer {layer_id}",
                )
                result = self._compare_tensor_pair(
                    lhs_layers[layer_id][tensor_key],
                    rhs_layers[layer_id][tensor_key],
                    atol=atol,
                )
                print(
                    f"[{lhs_name} vs {rhs_name}] layer={layer_id} tensor={tensor_key} "
                    f"allclose={result['allclose']} max_abs_diff={result['max_abs_diff']}"
                )
                if not result["allclose"]:
                    failure_messages.append(
                        " | ".join(
                            [
                                f"pair={lhs_name} vs {rhs_name}",
                                f"layer={layer_id}",
                                f"tensor={tensor_key}",
                                f"shape_lhs={result['shape_lhs']}",
                                f"shape_rhs={result['shape_rhs']}",
                                f"max_abs_diff={result['max_abs_diff']}",
                                f"first_bad_index={result['first_bad_index']}",
                                f"lhs_value={result['lhs_value']}",
                                f"rhs_value={result['rhs_value']}",
                            ]
                        )
                    )
        if failure_messages:
            config = (
                f"server_config(deterministic={DETERMINISTIC_MODE}, "
                f"attention_backend={ATTENTION_BACKEND or 'default'}, "
                f"disable_piecewise_cuda_graph={DISABLE_PIECEWISE_CUDA_GRAPH})"
            )
            self.fail("\n".join([config, *failure_messages]))

    def test_direct_vs_graft_vs_hf_kv_tensors(self):
        full_ids = self.tokenizer.encode(self.full_text, add_special_tokens=False)
        self.assertGreaterEqual(len(full_ids), 4, full_ids)
        split = max(1, len(full_ids) // 2)
        prefix_ids = full_ids[:split]
        suffix_ids = full_ids[split:]
        self.assertTrue(prefix_ids)
        self.assertTrue(suffix_ids)

        print(f"full_ids={full_ids}")
        print(f"prefix_ids={prefix_ids}")
        print(f"suffix_ids={suffix_ids}")

        direct_response = self._post_generate(
            {
                "input_ids": full_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "kv_export": {
                    "token_start": 0,
                    "token_end": len(full_ids),
                    "origin_start": 0,
                    "persist": True,
                    "ttl_seconds": 300,
                    "name": "tensor-equivalence-direct",
                },
            }
        )
        direct_export = self._extract_single_kv_export(direct_response)
        direct_handle = direct_export["handle"]
        self._track_handle(direct_handle)

        prefix_response = self._post_generate(
            {
                "input_ids": prefix_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "kv_export": {
                    "token_start": 0,
                    "token_end": len(prefix_ids),
                    "origin_start": 0,
                    "persist": True,
                    "ttl_seconds": 300,
                    "name": "tensor-equivalence-prefix",
                },
            }
        )
        prefix_export = self._extract_single_kv_export(prefix_response)
        prefix_handle = prefix_export["handle"]
        self._track_handle(prefix_handle)

        graft_response = self._post_generate(
            {
                "input_ids": suffix_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "kv_graft": {
                    "segments": [
                        {
                            "handle": prefix_handle,
                            "origin_start": 0,
                        }
                    ]
                },
                "kv_export": {
                    "token_start": 0,
                    "token_end": len(full_ids),
                    "origin_start": 0,
                    "persist": True,
                    "ttl_seconds": 300,
                    "name": "tensor-equivalence-graft",
                },
            }
        )
        graft_export = self._extract_single_kv_export(graft_response)
        graft_handle = graft_export["handle"]
        self._track_handle(graft_handle)

        self.assertEqual(direct_export["token_count"], len(full_ids))
        self.assertEqual(prefix_export["token_count"], len(prefix_ids))
        self.assertEqual(graft_export["token_count"], len(full_ids))

        direct_tensors = self._fetch_handle_tensors(direct_handle)
        graft_tensors = self._fetch_handle_tensors(graft_handle)

        self.assertEqual(direct_tensors["handle_meta"]["token_count"], len(full_ids))
        self.assertEqual(graft_tensors["handle_meta"]["token_count"], len(full_ids))
        self.assertEqual(len(direct_tensors["device_indices"]), len(full_ids))
        self.assertEqual(len(graft_tensors["device_indices"]), len(full_ids))
        self.assertEqual(direct_tensors["token_ids"], full_ids)
        self.assertEqual(graft_tensors["token_ids"], full_ids)

        direct_layers = self._server_layer_map(direct_tensors)
        graft_layers = self._server_layer_map(graft_tensors)
        self.assertEqual(sorted(direct_layers.keys()), sorted(graft_layers.keys()))

        sample_layer = next(iter(direct_layers.values()))
        if "k_nope" in sample_layer or "k_rope" in sample_layer:
            self.fail("MLA KV comparison against HF not implemented in this manual test")

        self._assert_layerwise_allclose(
            lhs_name="direct",
            rhs_name="graft",
            lhs_layers=direct_layers,
            rhs_layers=graft_layers,
            tensor_keys=("k", "v"),
            atol=self.direct_vs_graft_atol,
        )

        hf_layers = self._hf_mha_layer_map(full_ids)
        self.assertEqual(sorted(direct_layers.keys()), sorted(hf_layers.keys()))

        self._assert_layerwise_allclose(
            lhs_name="direct",
            rhs_name="hf",
            lhs_layers=direct_layers,
            rhs_layers=hf_layers,
            tensor_keys=("k", "v"),
            atol=self.hf_atol,
        )
        self._assert_layerwise_allclose(
            lhs_name="graft",
            rhs_name="hf",
            lhs_layers=graft_layers,
            rhs_layers=hf_layers,
            tensor_keys=("k", "v"),
            atol=self.hf_atol,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
