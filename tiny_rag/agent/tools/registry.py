from __future__ import annotations

import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from tiny_rag.cancellation import (
    CancellationToken,
    OperationCancelled,
    call_with_cancellation,
)

from .tool import FunctionDefinition, Tool, ToolExecutionError, ToolResult


toolErrorHint = "\n\n[Analyze the error above and try a different approach.]"
DefaultMaxToolOutput = 16000
headRatio = 0.7
truncationMarkerReserve = 200


@dataclass(frozen=True)
class ValidationError:
    param: str
    message: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._max_tool_output_size = 0
        self._pending_futures: dict[str, set[Future[Any]]] = {}
        self._deferred_cleanup: set[str] = set()
        self._pending_lock = threading.Lock()

    def set_max_tool_output_size(self, max_chars: int) -> None:
        self._max_tool_output_size = max_chars

    def _get_max_tool_output(self) -> int:
        if self._max_tool_output_size > 0:
            return self._max_tool_output_size
        return DefaultMaxToolOutput

    def register_tool(self, tool: Tool) -> None:
        name = tool.name()
        if name in self._tools:
            return
        self._tools[name] = tool

    def get_tool(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool not found: {name}") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_function_definitions(self) -> list[FunctionDefinition]:
        return [
            FunctionDefinition(
                name=self._tools[name].name(),
                description=self._tools[name].description(),
                parameters=self._tools[name].parameters(),
            )
            for name in self.list_tools()
        ]

    def execute_tool(
        self,
        name: str,
        args: Any,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        try:
            tool = self.get_tool(name)
        except KeyError as exc:
            return ToolResult(success=False, error=f"{exc.args[0]}{toolErrorHint}")

        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()

        casted_args = cast_params(args, tool.parameters())
        validation_errs = validate_params(casted_args, tool.parameters())
        if validation_errs:
            return ToolResult(
                success=False,
                error=format_validation_errors(validation_errs) + toolErrorHint,
            )

        try:
            result = _coerce_tool_result(
                call_with_cancellation(
                    tool.execute,
                    casted_args,
                    cancellation_token=cancellation_token,
                )
            )
        except OperationCancelled as exc:
            return ToolResult(success=False, error=str(exc) + toolErrorHint)
        except ToolExecutionError as exc:
            if exc.result is not None:
                return exc.result
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        max_output = self._get_max_tool_output()
        if result.output and len(result.output) > max_output:
            result.output = truncate_tool_output(result.output, max_output)

        if not result.success and result.error:
            result.error += toolErrorHint
        return result

    def track_background_tool_execution(self, name: str, future: Future[Any]) -> None:
        with self._pending_lock:
            self._pending_futures.setdefault(name, set()).add(future)
        future.add_done_callback(lambda _future: self._finish_background_tool_execution(name))

    def cleanup(self) -> None:
        for name, tool in self._tools.items():
            if self._has_pending_tool_execution(name):
                with self._pending_lock:
                    self._deferred_cleanup.add(name)
                continue
            cleanup = getattr(tool, "cleanup", None)
            if cleanup is not None:
                cleanup()

    def _has_pending_tool_execution(self, name: str) -> bool:
        with self._pending_lock:
            futures = self._pending_futures.get(name, set())
            done = {future for future in futures if future.done()}
            futures.difference_update(done)
            if not futures:
                self._pending_futures.pop(name, None)
            return bool(futures)

    def _finish_background_tool_execution(self, name: str) -> None:
        should_cleanup = False
        with self._pending_lock:
            futures = self._pending_futures.get(name, set())
            done = {future for future in futures if future.done()}
            futures.difference_update(done)
            if not futures:
                self._pending_futures.pop(name, None)
                should_cleanup = name in self._deferred_cleanup
                self._deferred_cleanup.discard(name)

        if should_cleanup:
            tool = self._tools.get(name)
            cleanup = getattr(tool, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    pass

    RegisterTool = register_tool
    GetTool = get_tool
    ListTools = list_tools
    GetFunctionDefinitions = get_function_definitions
    ExecuteTool = execute_tool
    Cleanup = cleanup


def _coerce_tool_result(raw: ToolResult | dict[str, Any]) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    if isinstance(raw, dict):
        return ToolResult(
            success=bool(raw.get("success")),
            output=str(raw.get("output") or ""),
            data=raw.get("data"),
            error=str(raw.get("error") or ""),
            images=list(raw.get("images") or []),
        )
    raise TypeError("tool returned unsupported result type")


def cast_params(args: Any, schema: Any) -> Any:
    if schema in (None, {}, b"", "") or args in (None, b"", ""):
        return args

    schema_def = _load_json_object(schema)
    if not schema_def:
        return args
    properties = schema_def.get("properties")
    if not isinstance(properties, dict) or len(properties) == 0:
        return args

    args_map = _load_json_object(args)
    if args_map is None:
        return args

    changed = False
    out = dict(args_map)
    for key, value in args_map.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        target_type = prop.get("type")
        if not isinstance(target_type, str) or target_type == "":
            continue
        new_value, did_cast = _cast_value(value, target_type)
        if did_cast:
            out[key] = new_value
            changed = True
    if not changed:
        return args
    return out


def _cast_value(value: Any, target_type: str) -> tuple[Any, bool]:
    if target_type == "array":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return parsed, True
            return [value], True

    if target_type == "boolean":
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in ("true", "1", "yes"):
                return True, True
            if lowered in ("false", "0", "no"):
                return False, True
        if _is_json_number(value):
            if float(value) == 0:
                return False, True
            if float(value) == 1:
                return True, True

    if target_type == "integer":
        if isinstance(value, str):
            try:
                return int(value, 10), True
            except ValueError:
                pass
        if _is_json_number(value) and float(value).is_integer():
            return int(value), True

    if target_type == "number":
        if isinstance(value, str):
            try:
                return float(value), True
            except ValueError:
                pass

    if target_type == "string":
        if isinstance(value, bool):
            return ("true" if value else "false"), True
        if _is_json_number(value):
            if isinstance(value, float) and value.is_integer():
                return str(int(value)), True
            return str(value), True

    return value, False


def validate_params(args: Any, schema: Any) -> list[ValidationError]:
    if schema in (None, {}, b"", "") or args in (None, b"", ""):
        return []

    schema_def = _load_json_object(schema)
    if not schema_def:
        return []
    properties = schema_def.get("properties")
    if not isinstance(properties, dict) or len(properties) == 0:
        return []

    args_map = _load_json_object(args)
    if args_map is None:
        return []

    errs: list[ValidationError] = []
    required = schema_def.get("required")
    if isinstance(required, list):
        for field in required:
            if not isinstance(field, str):
                continue
            if field not in args_map or args_map[field] is None:
                errs.append(
                    ValidationError(
                        param=field,
                        message=f"required parameter '{field}' is missing",
                    )
                )

    for key, value in args_map.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        errs.extend(_validate_property(key, value, prop))
    return errs


def _validate_property(name: str, value: Any, prop: dict[str, Any]) -> list[ValidationError]:
    if value is None:
        return []

    errs: list[ValidationError] = []
    target_type = prop.get("type")
    if isinstance(target_type, str) and target_type and not _check_type(value, target_type):
        return [
            ValidationError(
                param=name,
                message=f"parameter '{name}' should be type '{target_type}'",
            )
        ]

    enum = prop.get("enum")
    if isinstance(enum, list) and enum and not _is_in_enum(value, enum):
        allowed = ", ".join(str(item) for item in enum)
        errs.append(
            ValidationError(
                param=name,
                message=f"parameter '{name}' must be one of [{allowed}]",
            )
        )

    if target_type in ("number", "integer"):
        num_value = _to_float(value)
        if "minimum" in prop and num_value < float(prop["minimum"]):
            errs.append(
                ValidationError(
                    param=name,
                    message=f"parameter '{name}' must be >= {_format_number(prop['minimum'])}",
                )
            )
        if "maximum" in prop and num_value > float(prop["maximum"]):
            errs.append(
                ValidationError(
                    param=name,
                    message=f"parameter '{name}' must be <= {_format_number(prop['maximum'])}",
                )
            )

    if target_type == "string" and isinstance(value, str):
        if "minLength" in prop and len(value) < float(prop["minLength"]):
            errs.append(
                ValidationError(
                    param=name,
                    message=(
                        f"parameter '{name}' must have at least "
                        f"{int(prop['minLength'])} characters"
                    ),
                )
            )
        if "maxLength" in prop and len(value) > float(prop["maxLength"]):
            errs.append(
                ValidationError(
                    param=name,
                    message=(
                        f"parameter '{name}' must have at most "
                        f"{int(prop['maxLength'])} characters"
                    ),
                )
            )

    if target_type == "array" and isinstance(value, list):
        if "minItems" in prop and len(value) < int(prop["minItems"]):
            errs.append(
                ValidationError(
                    param=name,
                    message=(
                        f"parameter '{name}' must have at least "
                        f"{int(prop['minItems'])} items"
                    ),
                )
            )
        if "maxItems" in prop and len(value) > int(prop["maxItems"]):
            errs.append(
                ValidationError(
                    param=name,
                    message=(
                        f"parameter '{name}' must have at most "
                        f"{int(prop['maxItems'])} items"
                    ),
                )
            )

        item_schema = prop.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errs.extend(_validate_property(f"{name}[{index}]", item, item_schema))

    return errs


def _check_type(value: Any, target_type: str) -> bool:
    if target_type == "string":
        return isinstance(value, str)
    if target_type == "number":
        return _is_json_number(value)
    if target_type == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            or isinstance(value, float)
            and value.is_integer()
        )
    if target_type == "boolean":
        return isinstance(value, bool)
    if target_type == "array":
        return isinstance(value, list)
    if target_type == "object":
        return isinstance(value, dict)
    return True


def _is_in_enum(value: Any, enum: list[Any]) -> bool:
    return any(str(value) == str(item) for item in enum)


def _to_float(value: Any) -> float:
    if _is_json_number(value):
        return float(value)
    return 0.0


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def format_validation_errors(errs: list[ValidationError]) -> str:
    if not errs:
        return ""
    return "Parameter validation failed: " + "; ".join(err.message for err in errs)


def truncate_tool_output(output: str, max_chars: int) -> str:
    rune_count = len(output)
    if max_chars <= 0 or rune_count <= max_chars:
        return output

    usable = max_chars - truncationMarkerReserve
    if usable <= 0:
        return output[:max_chars]

    head_size = int(float(usable) * headRatio)
    tail_size = usable - head_size
    if tail_size <= 0:
        tail_size = 0

    marker = (
        f"\n\n... [output truncated: {rune_count} \u2192 {max_chars} chars, "
        f"showing first {head_size} + last {tail_size}] ...\n\n"
    )
    if tail_size == 0:
        return output[:head_size] + marker
    return output[:head_size] + marker + output[-tail_size:]


def _load_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return None
        if isinstance(decoded, dict):
            return decoded
    return None


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
