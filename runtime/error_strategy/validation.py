# Author : atg
# Time   :  2026/11/1 10:47

"""工具参数校验(文档 4.4.1 第 3 类:输出解析错误)

模型生成的工具参数可能缺字段、多字段、类型不对——
把无效调用挡在执行之前,并把逐条错误说明回传给模型,
让它修正后重新发起调用,比执行到一半崩溃便宜得多。
"""

from typing import Any, Dict, List, Tuple

from runtime.message_types import ToolResultBlock, ToolUseBlock
from tools.register import ToolRegistry

from .errors import ToolParameterValidationError

# JSON Schema 类型名 -> Python 类型(用于 isinstance 检查)
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_tool_parameters(
    tool_name: str,
    parameters: Dict[str, Any],
    schema: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """验证工具参数是否符合 Schema

    返回 (是否全部通过, 错误明细列表)。
    校验三项:必需参数存在、无未知参数、参数类型正确。
    """
    errors: List[str] = []

    # 检查必需参数
    required = schema.get("required", [])
    for param_name in required:
        if param_name not in parameters:
            errors.append(f"Missing required parameter: {param_name}")

    # 检查未知参数与参数类型
    properties = schema.get("properties", {})
    for param_name, param_value in parameters.items():
        if param_name not in properties:
            errors.append(f"Unknown parameter: {param_name}")
            continue

        expected_type = properties[param_name].get("type")
        python_type = _TYPE_MAP.get(expected_type)
        if python_type is None:
            # schema 里没写或不认识的类型:跳过类型检查(不做过度校验)
            continue
        # 细节:Python 里 bool 是 int 的子类,isinstance(True, int) 为 True,
        # 声明 integer 的参数不该接受布尔值,这里显式排除
        if expected_type == "integer" and isinstance(param_value, bool):
            errors.append(f"Parameter '{param_name}' should be integer, got boolean")
        elif not isinstance(param_value, python_type):
            errors.append(
                f"Parameter '{param_name}' should be {expected_type}, got {type(param_value).__name__}"
            )

    return len(errors) == 0, errors


async def execute_tool_validated(
    tool_use: ToolUseBlock,
    tool_registry: ToolRegistry,
    app_state=None,
) -> ToolResultBlock:
    """执行工具前先做参数验证

    验证失败时返回带逐条错误说明的错误块,不再执行工具;
    错误说明会作为"观察"回传给模型,模型据此修正参数重新调用。
    """
    tool = tool_registry.get(tool_use.name)
    if not tool:
        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=f"Tool '{tool_use.name}' not found",
            is_error=True,
            error_type="ToolNotFoundError",
        )

    # 项目 Tool 接口的方法名是 input_schema()(文档示例为 get_input_schema)
    is_valid, errors = validate_tool_parameters(tool_use.name, tool_use.input, tool.input_schema())
    if not is_valid:
        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content="Parameter validation failed:\n" + "\n".join(errors),
            is_error=True,
            error_type="ParameterValidationError",
        )

    # 参数有效,执行工具
    try:
        raw = await tool.call(tool_use.input)
        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=raw.content,
            is_error=not raw.success,
            error_type=raw.error_type,
        )
    except Exception as e:
        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=str(e),
            is_error=True,
            error_type=type(e).__name__,
        )


def raise_if_invalid(tool_name: str, parameters: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """校验不通过时直接抛 ToolParameterValidationError(供偏好异常风格的调用方使用)"""
    is_valid, errors = validate_tool_parameters(tool_name, parameters, schema)
    if not is_valid:
        raise ToolParameterValidationError(tool_name, errors)
