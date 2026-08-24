#!/usr/bin/env python3
"""Idempotently configure the SAM 3 and SAM 3D FC 3.0 functions.

The deploy identity is resolved exclusively through the Alibaba Cloud SDK's
default credential chain. This command never accepts AccessKey values as CLI
arguments and never writes credentials to its result file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Protocol


DEFAULT_INITIALIZER_COMMAND = ["/bin/sh", "/srv/scripts/fc_initializer.sh"]
DEFAULT_OSS_ENDPOINT = "https://oss-cn-shenzhen-internal.aliyuncs.com"
FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
REGION_PATTERN = re.compile(r"^[a-z0-9-]+$")
ROLE_ARN_PATTERN = re.compile(r"^acs:ram::[0-9]+:role/[A-Za-z0-9@._+-]+$")


class ConfigurationError(RuntimeError):
    """Raised when the requested or returned FC configuration is unsafe."""


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    region: str
    role_arn: str
    oss_bucket: str
    oss_prefix: str
    oss_endpoint: str
    mount_dir: str
    segmenter_function: str
    generator_function: str
    segmenter_image: str
    generator_image: str
    trigger_name: str
    trigger_auth: str
    acr_instance_id: str | None
    initializer_timeout: int
    function_timeout: int
    cpu: float
    memory_size: int
    gpu_type: str
    gpu_memory_size: int
    disk_size: int
    segmenter_provisioned_instances: int
    generator_provisioned_instances: int


class FCAPI(Protocol):
    def get_function(self, name: str) -> dict[str, Any] | None: ...

    def create_function(self, spec: dict[str, Any]) -> dict[str, Any]: ...

    def update_function(
        self,
        name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]: ...

    def wait_function(self, name: str) -> dict[str, Any]: ...

    def get_trigger(self, function_name: str, trigger_name: str) -> dict[str, Any] | None: ...

    def create_trigger(
        self,
        function_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]: ...

    def update_trigger(
        self,
        function_name: str,
        trigger_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]: ...

    def get_provision_config(self, function_name: str) -> dict[str, Any] | None: ...

    def put_provision_config(
        self,
        function_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]: ...

    def wait_provision_config(
        self,
        function_name: str,
        target: int,
    ) -> dict[str, Any]: ...


def _validate_config(config: DeploymentConfig) -> None:
    if not REGION_PATTERN.fullmatch(config.region):
        raise ConfigurationError(f"invalid FC region: {config.region}")
    if not ROLE_ARN_PATTERN.fullmatch(config.role_arn):
        raise ConfigurationError("role ARN must be an acs:ram role ARN")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", config.oss_bucket):
        raise ConfigurationError("invalid OSS bucket name")
    if not config.oss_prefix or config.oss_prefix.startswith("/"):
        raise ConfigurationError("OSS prefix must be non-empty and omit the leading slash")
    if ".." in Path(config.oss_prefix).parts:
        raise ConfigurationError("OSS prefix must not contain parent traversal")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", config.oss_prefix):
        raise ConfigurationError("OSS prefix contains unsupported characters")
    expected_oss_endpoint = f"https://oss-{config.region}-internal.aliyuncs.com"
    if config.oss_endpoint != expected_oss_endpoint:
        raise ConfigurationError(
            f"OSS endpoint must use the regional internal endpoint: {expected_oss_endpoint}"
        )
    if not config.mount_dir.startswith("/") or config.mount_dir == "/":
        raise ConfigurationError("mount directory must be a non-root absolute path")
    if ".." in Path(config.mount_dir).parts:
        raise ConfigurationError("mount directory must not contain parent traversal")
    for label, name in (
        ("segmenter", config.segmenter_function),
        ("generator", config.generator_function),
    ):
        if not FUNCTION_NAME_PATTERN.fullmatch(name):
            raise ConfigurationError(f"invalid {label} function name: {name}")
    if config.segmenter_function == config.generator_function:
        raise ConfigurationError("segmenter and generator function names must differ")
    if not FUNCTION_NAME_PATTERN.fullmatch(config.trigger_name):
        raise ConfigurationError(f"invalid trigger name: {config.trigger_name}")
    for label, image in (
        ("segmenter", config.segmenter_image),
        ("generator", config.generator_image),
    ):
        if image.startswith(("http://", "https://")) or ":" not in image:
            raise ConfigurationError(f"{label} image must be a tagged registry reference")
        if any(character.isspace() for character in image):
            raise ConfigurationError(f"{label} image contains whitespace")
    if config.trigger_auth not in {"anonymous", "function"}:
        raise ConfigurationError("trigger auth must be anonymous or function")
    if config.acr_instance_id and any(
        character.isspace() for character in config.acr_instance_id
    ):
        raise ConfigurationError("ACR instance ID contains whitespace")
    if not 1 <= config.initializer_timeout <= 300:
        raise ConfigurationError("initializer timeout must be between 1 and 300 seconds")
    if config.function_timeout < config.initializer_timeout:
        raise ConfigurationError("function timeout must not be shorter than initializer timeout")
    if config.cpu <= 0 or config.memory_size <= 0 or config.gpu_memory_size <= 0:
        raise ConfigurationError("FC CPU, memory, and GPU memory must be positive")
    if config.disk_size not in {512, 10240}:
        raise ConfigurationError("disk size must be 512 or 10240 MB")
    for label, target in (
        ("segmenter", config.segmenter_provisioned_instances),
        ("generator", config.generator_provisioned_instances),
    ):
        if (
            not isinstance(target, int)
            or isinstance(target, bool)
            or not 0 <= target <= 100
        ):
            raise ConfigurationError(
                f"{label} provisioned instances must be an integer between 0 and 100"
            )


def _common_environment() -> dict[str, str]:
    return {
        "CORS_ALLOW_ORIGINS": "*",
        "FC_INITIALIZER_HTTP_TIMEOUT": "295",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "KEEP_ALIVE_TIMEOUT": "900",
        "PORT": "9000",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _segmenter_environment() -> dict[str, str]:
    environment = _common_environment()
    environment.update(
        {
            "SAM3_CHECKPOINT_PATH": "/mnt/nas/sam3d/sam3/sam3.pt",
            "SAM3_MAX_IMAGE_PIXELS": "40000000",
            "SAM3_MAX_POINTS": "64",
            "SAM3_MAX_UPLOAD_MB": "20",
            "SAM3_ROOT": "/opt/sam3",
        }
    )
    return environment


def _generator_environment() -> dict[str, str]:
    environment = _common_environment()
    environment.update(
        {
            "ATTN_BACKEND": "sdpa",
            "HF_HOME": "/mnt/nas/sam3d/cache/huggingface",
            "LIDRA_SKIP_INIT": "true",
            "SAM3D_COMPILE": "false",
            "SAM3D_CONFIG_PATH": "/mnt/nas/sam3d/hf/pipeline.yaml",
            "SAM3D_MAX_IMAGE_PIXELS": "40000000",
            "SAM3D_MAX_REQUEST_MB": "30",
            "SAM3D_MAX_UPLOAD_MB": "20",
            "SAM3D_ROOT": "/opt/sam-3d-objects",
            "SAM3D_TMP_DIR": "/tmp/sam3d",
            "SPARSE_ATTN_BACKEND": "sdpa",
            "SPARSE_BACKEND": "spconv",
            "TORCH_HOME": "/mnt/nas/sam3d/cache/torch",
        }
    )
    return environment


def build_function_spec(
    config: DeploymentConfig,
    *,
    function_name: str,
    image: str,
    environment: dict[str, str],
    description: str,
) -> dict[str, Any]:
    """Return the FC 2023-03-30 CreateFunction body as plain JSON data."""

    _validate_config(config)
    container_config: dict[str, Any] = {
        "accelerationType": "Default",
        "healthCheckConfig": {
            "failureThreshold": 3,
            "httpGetUrl": "/healthz",
            "initialDelaySeconds": 1,
            "periodSeconds": 5,
            "successThreshold": 1,
            "timeoutSeconds": 2,
        },
        "image": image,
        "port": 9000,
    }
    if config.acr_instance_id:
        container_config["acrInstanceId"] = config.acr_instance_id

    return {
        "cpu": config.cpu,
        "customContainerConfig": container_config,
        "description": description,
        "diskSize": config.disk_size,
        "environmentVariables": environment,
        "functionName": function_name,
        "gpuConfig": {
            "gpuMemorySize": config.gpu_memory_size,
            "gpuType": config.gpu_type,
        },
        "handler": "index.handler",
        "instanceConcurrency": 1,
        "instanceLifecycleConfig": {
            "initializer": {
                "command": list(DEFAULT_INITIALIZER_COMMAND),
                "timeout": config.initializer_timeout,
            }
        },
        "internetAccess": False,
        "memorySize": config.memory_size,
        "ossMountConfig": {
            "mountPoints": [
                {
                    "bucketName": config.oss_bucket,
                    "bucketPath": f"/{config.oss_prefix}",
                    "endpoint": config.oss_endpoint,
                    "mountDir": config.mount_dir,
                    "readOnly": True,
                }
            ]
        },
        "role": config.role_arn,
        "runtime": "custom-container",
        "timeout": config.function_timeout,
    }


def build_deployment_plan(config: DeploymentConfig) -> dict[str, Any]:
    _validate_config(config)
    trigger_config = json.dumps(
        {
            "authType": config.trigger_auth,
            "disableURLInternet": False,
            "methods": ["GET", "POST", "OPTIONS"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    trigger = {
        "description": "Browser/API HTTP entrypoint managed by deploy_from_hk.sh",
        "qualifier": "LATEST",
        "triggerConfig": trigger_config,
        "triggerName": config.trigger_name,
        "triggerType": "http",
    }
    provision_specs = {
        "segmenter": {
            "alwaysAllocateCPU": True,
            "alwaysAllocateGPU": True,
            "defaultTarget": config.segmenter_provisioned_instances,
        },
        "generator": {
            "alwaysAllocateCPU": True,
            "alwaysAllocateGPU": True,
            "defaultTarget": config.generator_provisioned_instances,
        },
    }
    return {
        "apiVersion": "2023-03-30",
        "region": config.region,
        "functions": [
            {
                "kind": "segmenter",
                "spec": build_function_spec(
                    config,
                    function_name=config.segmenter_function,
                    image=config.segmenter_image,
                    environment=_segmenter_environment(),
                    description="SAM 3 point-prompt mask service",
                ),
                "provision": provision_specs["segmenter"],
                "trigger": trigger,
            },
            {
                "kind": "generator",
                "spec": build_function_spec(
                    config,
                    function_name=config.generator_function,
                    image=config.generator_image,
                    environment=_generator_environment(),
                    description="SAM 3D Objects image-and-mask generation service",
                ),
                "provision": provision_specs["generator"],
                "trigger": trigger,
            },
        ],
    }


def _mismatches(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path or '<root>'}: expected object"]
        errors: list[str] = []
        for key, value in expected.items():
            child_path = f"{path}.{key}" if path else key
            if key not in actual:
                errors.append(f"{child_path}: missing")
            else:
                errors.extend(_mismatches(value, actual[key], child_path))
        return errors
    if isinstance(expected, list):
        if expected != actual:
            return [f"{path}: expected {expected!r}, got {actual!r}"]
        return []
    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def _function_update_spec(create_spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in create_spec.items() if key != "functionName"}


def _trigger_update_spec(create_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in create_spec.items()
        if key not in {"triggerName", "triggerType"}
    }


def reconcile_deployment(config: DeploymentConfig, api: FCAPI) -> dict[str, Any]:
    """Create or update both functions and triggers, then read them back."""

    plan = build_deployment_plan(config)
    results: list[dict[str, Any]] = []

    for target in plan["functions"]:
        create_spec = target["spec"]
        function_name = create_spec["functionName"]
        existing = api.get_function(function_name)
        if existing is None:
            api.create_function(create_spec)
            function_action = "created"
        else:
            desired_update = _function_update_spec(create_spec)
            if _mismatches(desired_update, existing):
                api.update_function(function_name, desired_update)
                function_action = "updated"
            else:
                function_action = "unchanged"

        actual_function = api.wait_function(function_name)
        function_errors = _mismatches(create_spec, actual_function)
        if function_errors:
            raise ConfigurationError(
                f"FC function readback mismatch for {function_name}: "
                + "; ".join(function_errors[:10])
            )

        trigger_create = target["trigger"]
        trigger_name = trigger_create["triggerName"]
        desired_trigger_update = _trigger_update_spec(trigger_create)
        expected_config = json.loads(desired_trigger_update["triggerConfig"])
        existing_trigger = api.get_trigger(function_name, trigger_name)
        if existing_trigger is None:
            api.create_trigger(function_name, trigger_create)
            trigger_action = "created"
        else:
            trigger_probe = dict(existing_trigger)
            actual_config = json.loads(trigger_probe.get("triggerConfig", "{}"))
            trigger_probe["triggerConfig"] = json.dumps(
                actual_config,
                separators=(",", ":"),
                sort_keys=True,
            )
            if _mismatches(desired_trigger_update, trigger_probe):
                api.update_trigger(function_name, trigger_name, desired_trigger_update)
                trigger_action = "updated"
            else:
                trigger_action = "unchanged"

        actual_trigger = api.get_trigger(function_name, trigger_name)
        if actual_trigger is None:
            raise ConfigurationError(f"FC trigger disappeared after write: {function_name}")
        actual_trigger_config = json.loads(actual_trigger.get("triggerConfig", "{}"))
        if actual_trigger_config != expected_config:
            raise ConfigurationError(
                f"FC trigger readback mismatch for {function_name}: "
                f"expected {expected_config!r}, got {actual_trigger_config!r}"
            )

        http_trigger = actual_trigger.get("httpTrigger") or {}

        provision_spec = target["provision"]
        provision_target = int(provision_spec["defaultTarget"])
        existing_provision = api.get_provision_config(function_name)
        if existing_provision is None:
            api.put_provision_config(function_name, provision_spec)
            provision_action = "created"
        elif function_action != "unchanged" or _mismatches(
            provision_spec, existing_provision
        ):
            # Re-apply an unchanged provision target after a function/image
            # update so FC has an explicit synchronization point for the new
            # LATEST instances and their Initializer execution.
            api.put_provision_config(function_name, provision_spec)
            provision_action = "updated"
        else:
            provision_action = "unchanged"

        if provision_target > 0:
            actual_provision = api.wait_provision_config(function_name, provision_target)
            provision_ready = True
            wait_skipped = False
        else:
            actual_provision = api.get_provision_config(function_name) or dict(provision_spec)
            provision_ready = False
            wait_skipped = True

        provision_errors = _mismatches(provision_spec, actual_provision)
        if provision_errors:
            raise ConfigurationError(
                f"FC provision config readback mismatch for {function_name}: "
                + "; ".join(provision_errors[:10])
            )
        current_error = str(actual_provision.get("currentError") or "")
        current = actual_provision.get("current")
        if provision_target > 0:
            if current_error:
                raise ConfigurationError(
                    f"FC provisioned instance failed for {function_name}: {current_error}"
                )
            if not isinstance(current, int) or isinstance(current, bool) or current < provision_target:
                raise ConfigurationError(
                    f"FC provisioned instance readback is not ready for {function_name}: "
                    f"current={current!r}, target={provision_target}"
                )

        results.append(
            {
                "functionAction": function_action,
                "functionName": function_name,
                "kind": target["kind"],
                "provision": {
                    "action": provision_action,
                    "alwaysAllocateCPU": actual_provision.get("alwaysAllocateCPU"),
                    "alwaysAllocateGPU": actual_provision.get("alwaysAllocateGPU"),
                    "current": current,
                    "currentError": current_error or None,
                    "defaultTarget": provision_target,
                    "ready": provision_ready,
                    "waitSkipped": wait_skipped,
                },
                "resolvedImageUri": (actual_function.get("customContainerConfig") or {}).get(
                    "resolvedImageUri"
                ),
                "triggerAction": trigger_action,
                "triggerName": trigger_name,
                "urlInternet": http_trigger.get("urlInternet"),
                "verified": True,
            }
        )

    return {
        "apiVersion": plan["apiVersion"],
        "functions": results,
        "region": config.region,
    }


def _error_code(error: Exception) -> str:
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if value is not None:
            return str(value)
    data = getattr(error, "data", None)
    if isinstance(data, dict):
        for key in ("Code", "code", "statusCode"):
            if key in data:
                return str(data[key])
    return ""


def _is_not_found(error: Exception) -> bool:
    code = _error_code(error).lower()
    message = str(error).lower()
    return code == "404" or code.endswith("notfound") or (
        "not found" in message and "permission" not in message
    )


class AlibabaFCAPI:
    """Thin plain-dict adapter around alibabacloud_fc20230330."""

    def __init__(self, region: str, *, wait_timeout: int = 900) -> None:
        try:
            from alibabacloud_credentials.client import Client as CredentialClient
            from alibabacloud_fc20230330.client import Client as FCClient
            from alibabacloud_fc20230330 import models
            from alibabacloud_tea_openapi import models as open_api_models
        except ImportError as exc:
            raise ConfigurationError(
                "FC SDK is unavailable; install alibabacloud-fc20230330 and "
                "alibabacloud-credentials in the deployment tool environment"
            ) from exc

        credential = CredentialClient()
        sdk_config = open_api_models.Config(credential=credential)
        sdk_config.endpoint = f"fcv3.{region}.aliyuncs.com"
        self._client = FCClient(sdk_config)
        self._models = models
        self._wait_timeout = wait_timeout

    @staticmethod
    def _map(body: Any) -> dict[str, Any]:
        if body is None:
            return {}
        if hasattr(body, "to_map"):
            return body.to_map()
        if isinstance(body, dict):
            return body
        raise ConfigurationError(f"unexpected FC SDK response body: {type(body).__name__}")

    def get_function(self, name: str) -> dict[str, Any] | None:
        try:
            response = self._client.get_function(
                name,
                self._models.GetFunctionRequest(),
            )
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return self._map(response.body)

    def create_function(self, spec: dict[str, Any]) -> dict[str, Any]:
        body = self._models.CreateFunctionInput().from_map(spec)
        request = self._models.CreateFunctionRequest(body=body)
        return self._map(self._client.create_function(request).body)

    def update_function(
        self,
        name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._models.UpdateFunctionInput().from_map(spec)
        request = self._models.UpdateFunctionRequest(body=body)
        return self._map(self._client.update_function(name, request).body)

    def wait_function(self, name: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._wait_timeout
        while True:
            current = self.get_function(name)
            if current is None:
                raise ConfigurationError(f"FC function not found during readback: {name}")
            state = str(current.get("state") or "")
            update_status = str(current.get("lastUpdateStatus") or "")
            if state == "Failed" or update_status == "Failed":
                reason = current.get("lastUpdateStatusReason") or current.get("stateReason")
                raise ConfigurationError(f"FC function deployment failed for {name}: {reason}")
            state_ready = state in {"", "Active"}
            update_ready = update_status in {"", "Successful"}
            if state_ready and update_ready:
                return current
            if time.monotonic() >= deadline:
                raise ConfigurationError(
                    f"timed out waiting for FC function {name}: "
                    f"state={state!r}, lastUpdateStatus={update_status!r}"
                )
            time.sleep(5)

    def get_trigger(self, function_name: str, trigger_name: str) -> dict[str, Any] | None:
        try:
            response = self._client.get_trigger(function_name, trigger_name)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return self._map(response.body)

    def create_trigger(
        self,
        function_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._models.CreateTriggerInput().from_map(spec)
        request = self._models.CreateTriggerRequest(body=body)
        return self._map(self._client.create_trigger(function_name, request).body)

    def update_trigger(
        self,
        function_name: str,
        trigger_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._models.UpdateTriggerInput().from_map(spec)
        request = self._models.UpdateTriggerRequest(body=body)
        return self._map(
            self._client.update_trigger(function_name, trigger_name, request).body
        )

    def get_provision_config(self, function_name: str) -> dict[str, Any] | None:
        request = self._models.GetProvisionConfigRequest(qualifier="LATEST")
        try:
            response = self._client.get_provision_config(function_name, request)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return self._map(response.body)

    def put_provision_config(
        self,
        function_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._models.PutProvisionConfigInput().from_map(spec)
        request = self._models.PutProvisionConfigRequest(
            body=body,
            qualifier="LATEST",
        )
        return self._map(
            self._client.put_provision_config(function_name, request).body
        )

    def wait_provision_config(
        self,
        function_name: str,
        target: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._wait_timeout
        while True:
            current = self.get_provision_config(function_name)
            if current is None:
                raise ConfigurationError(
                    f"FC provision config not found during readback: {function_name}"
                )
            count = current.get("current")
            current_error = str(current.get("currentError") or "")
            if (
                isinstance(count, int)
                and not isinstance(count, bool)
                and count >= target
                and not current_error
            ):
                return current
            if time.monotonic() >= deadline:
                raise ConfigurationError(
                    f"timed out waiting for FC provisioned instances for {function_name}: "
                    f"current={count!r}, target={target}, currentError={current_error!r}"
                )
            time.sleep(5)


def _write_result(path: Path, result: dict[str, Any]) -> None:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ConfigurationError(f"result path must not be a symlink: {requested}")
    destination = requested.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        raise ConfigurationError(f"result path must be a regular file path: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update and verify the SAM3 segmenter and SAM3D generator FC functions."
    )
    parser.add_argument("--region", default="cn-shenzhen")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--oss-bucket", required=True)
    parser.add_argument("--oss-prefix", required=True)
    parser.add_argument("--oss-endpoint", default=DEFAULT_OSS_ENDPOINT)
    parser.add_argument("--mount-dir", default="/mnt/nas/sam3d")
    parser.add_argument("--segmenter-function", default="sam3-segmenter")
    parser.add_argument("--generator-function", default="sam3d-generator")
    parser.add_argument("--segmenter-image", required=True)
    parser.add_argument("--generator-image", required=True)
    parser.add_argument("--trigger-name", default="http-trigger")
    parser.add_argument("--trigger-auth", choices=("anonymous", "function"), default="anonymous")
    parser.add_argument("--acr-instance-id")
    parser.add_argument("--initializer-timeout", type=int, default=300)
    parser.add_argument("--function-timeout", type=int, default=1800)
    parser.add_argument("--cpu", type=float, default=8)
    parser.add_argument("--memory-size", type=int, default=65536)
    parser.add_argument("--gpu-type", default="fc.gpu.ada.1")
    parser.add_argument("--gpu-memory-size", type=int, default=49152)
    parser.add_argument("--disk-size", type=int, default=10240)
    parser.add_argument("--segmenter-provisioned-instances", type=int, default=1)
    parser.add_argument("--generator-provisioned-instances", type=int, default=1)
    parser.add_argument("--wait-timeout", type=int, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the complete non-secret plan without importing the SDK or changing cloud state",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> DeploymentConfig:
    return DeploymentConfig(
        region=args.region,
        role_arn=args.role_arn,
        oss_bucket=args.oss_bucket,
        oss_prefix=args.oss_prefix.strip("/"),
        oss_endpoint=args.oss_endpoint,
        mount_dir=args.mount_dir,
        segmenter_function=args.segmenter_function,
        generator_function=args.generator_function,
        segmenter_image=args.segmenter_image,
        generator_image=args.generator_image,
        trigger_name=args.trigger_name,
        trigger_auth=args.trigger_auth,
        acr_instance_id=args.acr_instance_id,
        initializer_timeout=args.initializer_timeout,
        function_timeout=args.function_timeout,
        cpu=args.cpu,
        memory_size=args.memory_size,
        gpu_type=args.gpu_type,
        gpu_memory_size=args.gpu_memory_size,
        disk_size=args.disk_size,
        segmenter_provisioned_instances=args.segmenter_provisioned_instances,
        generator_provisioned_instances=args.generator_provisioned_instances,
    )


def main() -> int:
    args = _parse_args()
    try:
        config = _config_from_args(args)
        if args.dry_run:
            result = build_deployment_plan(config)
        else:
            api = AlibabaFCAPI(config.region, wait_timeout=args.wait_timeout)
            result = reconcile_deployment(config, api)
        if args.output:
            _write_result(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (ConfigurationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 1
    except Exception as exc:  # Generated SDK exceptions do not share a stable base class.
        print(f"error: FC SDK request failed: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
