"""Offline contract tests for the FC 2023-03-30 deployment helper."""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "configure_fc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("configure_fc", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fc = _load_module()


def _config(**overrides):
    values = {
        "region": "cn-shenzhen",
        "role_arn": "acs:ram::1234567890123456:role/sam3d-fc-runtime",
        "oss_bucket": "example-bucket",
        "oss_prefix": "sam3d/releases/bundle-deadbeef1234",
        "oss_endpoint": "https://oss-cn-shenzhen-internal.aliyuncs.com",
        "mount_dir": "/mnt/nas/sam3d",
        "function_name": "sam3d-object",
        "image": "registry-vpc.cn-shenzhen.aliyuncs.com/ns/repo:sam3-sam3d-deadbeef",
        "trigger_name": "http-trigger",
        "trigger_auth": "anonymous",
        "acr_instance_id": None,
        "initializer_timeout": 300,
        "function_timeout": 1800,
        "cpu": 8.0,
        "memory_size": 65536,
        "gpu_type": "fc.gpu.ada.1",
        "gpu_memory_size": 49152,
        "disk_size": 10240,
        "provisioned_instances": 0,
        "reserved_concurrency": 1,
    }
    values.update(overrides)
    return fc.DeploymentConfig(**values)


class FakeFCAPI:
    def __init__(self) -> None:
        self.functions = {}
        self.triggers = {}
        self.provisions = {}
        self.concurrency = {}
        self.function_creates = 0
        self.function_updates = 0
        self.trigger_creates = 0
        self.trigger_updates = 0
        self.provision_puts = 0
        self.provision_waits = 0
        self.concurrency_puts = 0

    def get_function(self, name):
        value = self.functions.get(name)
        return copy.deepcopy(value) if value is not None else None

    def create_function(self, spec):
        self.function_creates += 1
        value = copy.deepcopy(spec)
        value["state"] = "Active"
        value["lastUpdateStatus"] = "Successful"
        value["customContainerConfig"]["resolvedImageUri"] = spec[
            "customContainerConfig"
        ]["image"]
        self.functions[spec["functionName"]] = value
        return copy.deepcopy(value)

    def update_function(self, name, spec):
        self.function_updates += 1
        self.functions[name].update(copy.deepcopy(spec))
        return copy.deepcopy(self.functions[name])

    def wait_function(self, name):
        return copy.deepcopy(self.functions[name])

    def get_trigger(self, function_name, trigger_name):
        value = self.triggers.get((function_name, trigger_name))
        return copy.deepcopy(value) if value is not None else None

    def create_trigger(self, function_name, spec):
        self.trigger_creates += 1
        value = copy.deepcopy(spec)
        value["httpTrigger"] = {
            "urlInternet": f"https://{function_name}.example.fc.aliyuncs.com"
        }
        self.triggers[(function_name, spec["triggerName"])] = value
        return copy.deepcopy(value)

    def update_trigger(self, function_name, trigger_name, spec):
        self.trigger_updates += 1
        self.triggers[(function_name, trigger_name)].update(copy.deepcopy(spec))
        return copy.deepcopy(self.triggers[(function_name, trigger_name)])

    def get_provision_config(self, function_name):
        value = self.provisions.get(function_name)
        return copy.deepcopy(value) if value is not None else None

    def put_provision_config(self, function_name, spec):
        self.provision_puts += 1
        value = copy.deepcopy(spec)
        value["current"] = spec["defaultTarget"]
        value["currentError"] = ""
        self.provisions[function_name] = value
        return copy.deepcopy(value)

    def wait_provision_config(self, function_name, target):
        self.provision_waits += 1
        value = self.provisions[function_name]
        if value["current"] < target or value["currentError"]:
            raise AssertionError("fake provision config is not ready")
        return copy.deepcopy(value)

    def get_concurrency_config(self, function_name):
        value = self.concurrency.get(function_name)
        return copy.deepcopy(value) if value is not None else None

    def put_concurrency_config(self, function_name, spec):
        self.concurrency_puts += 1
        self.concurrency[function_name] = copy.deepcopy(spec)
        return copy.deepcopy(spec)


class FCConfigureTests(unittest.TestCase):
    def test_plan_has_one_combined_image_initializer_mount_and_single_gpu_limit(self) -> None:
        config = _config()
        plan = fc.build_deployment_plan(config)

        self.assertEqual(plan["apiVersion"], "2023-03-30")
        self.assertEqual(len(plan["functions"]), 1)
        target = plan["functions"][0]
        self.assertEqual(target["kind"], "unified")
        self.assertEqual(target["concurrency"], {"reservedConcurrency": 1})
        initializer = target["spec"]["instanceLifecycleConfig"]["initializer"]
        self.assertEqual(
            initializer["command"],
            ["/bin/sh", "/srv/scripts/fc_initializer.sh"],
        )
        self.assertEqual(initializer["timeout"], 300)
        self.assertEqual(target["spec"]["instanceConcurrency"], 1)
        mount = target["spec"]["ossMountConfig"]["mountPoints"][0]
        self.assertTrue(mount["readOnly"])
        self.assertEqual(mount["mountDir"], "/mnt/nas/sam3d")
        self.assertEqual(
            target["provision"],
            {
                "alwaysAllocateCPU": True,
                "alwaysAllocateGPU": True,
                "defaultTarget": 0,
            },
        )
        trigger = json.loads(target["trigger"]["triggerConfig"])
        self.assertEqual(trigger["authType"], "anonymous")
        self.assertEqual(trigger["methods"], ["GET", "POST", "OPTIONS"])
        self.assertFalse(trigger["disableURLInternet"])

        environment = target["spec"]["environmentVariables"]
        self.assertEqual(environment["CORS_ALLOW_ORIGINS"], "*")
        self.assertEqual(
            environment["SAM3_CHECKPOINT_PATH"],
            "/mnt/nas/sam3d/sam3/sam3.pt",
        )
        self.assertEqual(
            environment["SAM3D_CONFIG_PATH"],
            "/mnt/nas/sam3d/hf/pipeline.yaml",
        )
        self.assertEqual(environment["GPU_LOCK_PATH"], "/tmp/sam3d-gpu.lock")
        self.assertNotIn("SAM3_PYTHON", environment)

    def test_reconcile_is_idempotent_and_waits_for_combined_initializer(self) -> None:
        api = FakeFCAPI()
        config = _config(provisioned_instances=1)

        first = fc.reconcile_deployment(config, api)
        second = fc.reconcile_deployment(config, api)

        self.assertEqual(api.function_creates, 1)
        self.assertEqual(api.function_updates, 0)
        self.assertEqual(api.trigger_creates, 1)
        self.assertEqual(api.trigger_updates, 0)
        self.assertEqual(api.provision_puts, 1)
        self.assertEqual(api.provision_waits, 2)
        self.assertEqual(api.concurrency_puts, 1)
        self.assertTrue(all(item["verified"] for item in first["functions"]))
        self.assertTrue(all(item["provision"]["ready"] for item in first["functions"]))
        self.assertTrue(
            all(item["functionAction"] == "unchanged" for item in second["functions"])
        )
        self.assertTrue(
            all(item["provision"]["action"] == "unchanged" for item in second["functions"])
        )
        self.assertTrue(all(item["urlInternet"] for item in second["functions"]))

    def test_function_image_update_reapplies_provision_before_warmup_readback(self) -> None:
        api = FakeFCAPI()
        config = _config()
        fc.reconcile_deployment(config, api)
        updated = replace(
            config,
            image="registry-vpc.cn-shenzhen.aliyuncs.com/ns/repo:sam3-sam3d-new",
        )

        result = fc.reconcile_deployment(updated, api)

        unified = result["functions"][0]
        self.assertEqual(unified["functionAction"], "updated")
        self.assertEqual(unified["provision"]["action"], "updated")
        self.assertEqual(api.provision_puts, 2)

    def test_zero_provision_target_configures_but_skips_warmup_wait(self) -> None:
        api = FakeFCAPI()
        config = _config(provisioned_instances=0)

        result = fc.reconcile_deployment(config, api)

        self.assertEqual(api.provision_puts, 1)
        self.assertEqual(api.provision_waits, 0)
        self.assertTrue(
            result["functions"][0]["provision"]["waitSkipped"]
        )
        self.assertTrue(
            not result["functions"][0]["provision"]["ready"]
        )

    def test_sdk_adapter_polls_until_current_reaches_target_without_error(self) -> None:
        api = fc.AlibabaFCAPI.__new__(fc.AlibabaFCAPI)
        api._wait_timeout = 1
        states = iter(
            [
                {"current": 0, "currentError": "initializer is still starting"},
                {"current": 1, "currentError": ""},
            ]
        )
        api.get_provision_config = mock.Mock(side_effect=lambda _name: next(states))

        with mock.patch.object(fc.time, "sleep", return_value=None):
            result = api.wait_provision_config("sam3d-object", 1)

        self.assertEqual(result["current"], 1)
        self.assertEqual(result["currentError"], "")
        self.assertEqual(api.get_provision_config.call_count, 2)

    def test_dry_run_never_imports_sdk_or_reads_credentials(self) -> None:
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith("ALIBABA_CLOUD_") or name.startswith("ALICLOUD_"):
                environment.pop(name)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dry-run",
                    "--region",
                    "cn-shenzhen",
                    "--role-arn",
                    "acs:ram::1234567890123456:role/sam3d-fc-runtime",
                    "--oss-bucket",
                    "example-bucket",
                    "--oss-prefix",
                    "sam3d/releases/bundle-deadbeef1234",
                    "--image",
                    "registry-vpc.cn-shenzhen.aliyuncs.com/ns/repo:sam3-sam3d-deadbeef",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["provision"]["defaultTarget"] for item in plan["functions"]],
                [0],
            )
            self.assertEqual(plan["functions"][0]["concurrency"]["reservedConcurrency"], 1)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_result_writer_refuses_symlink_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("unchanged\n", encoding="utf-8")
            link = root / "result.json"
            link.symlink_to(target)

            with self.assertRaisesRegex(fc.ConfigurationError, "symlink"):
                fc._write_result(link, {"secret": "must-not-write"})

            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged\n")


if __name__ == "__main__":
    unittest.main()
