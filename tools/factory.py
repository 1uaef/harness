# Author : atg
# Time   :  2026/11/2 18:22

"""build_tool() 工厂函数(文档 5.1.4)

按规范字典动态构造工具,支持参数化配置与延迟初始化:

    tool_spec = {
        "type": "...",            # 工具类型
        "name": "tool_name",      # 可选,仅描述用
        "config": {...},          # 工具特定的构造参数
    }
    tool = build_tool(tool_spec)

与文档示例的差异:文档用 if/elif 硬编码 bash/file_copy/database_query
等分支,依赖 5.1.3 的具体工具类(本项目按要求不实现例子);
这里改为**可扩展的 builder 注册表**——各业务方用 @tool_builder("类型")
注册自己的构造函数,工厂本身不依赖任何具体工具。

"custom" 分支保留文档的安全约束:只允许从受信任命名空间动态加载
工具类,防止任意 import 沦为代码注入入口。生产环境此处应抛类型化的
ToolExecutionError(见 runtime/error_strategy),这里用 ValueError
避免 tools 层反向依赖 runtime 层。
"""

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

from tools.register import ToolRegistry
from tools.tool import Tool

# builder 签名: builder(config) -> Tool(实例,config 为构造参数)
ToolBuilder = Callable[[Dict[str, Any]], Tool]

# 全局 builder 注册表:tool_type -> 构造函数
_BUILDERS: Dict[str, ToolBuilder] = {}


def tool_builder(tool_type: str) -> Callable[[ToolBuilder], ToolBuilder]:
    """装饰器:把构造函数注册为某类型工具的 builder

    用法:
        @tool_builder("database_query")
        def build_database_query(config: dict) -> Tool:
            return DatabaseQueryTool(connection_string=config["connection_string"])
    """

    def decorator(builder: ToolBuilder) -> ToolBuilder:
        _BUILDERS[tool_type] = builder
        return builder

    return decorator


def register_builder(tool_type: str, builder: ToolBuilder) -> None:
    """非装饰器形式的 builder 注册入口"""
    _BUILDERS[tool_type] = builder


def build_tool(
    tool_spec: Dict[str, Any],
    builders: Optional[Dict[str, ToolBuilder]] = None,
    allowed_prefixes: Tuple[str, ...] = ("my_harness_tools.",),
) -> Tool:
    """工厂函数:根据规范构造工具实例

    Args:
        tool_spec: 工具规范 {"type": ..., "config": {...}}
        builders: 调用方临时提供的 builder 映射(优先级高于全局注册表)
        allowed_prefixes: custom 类型允许动态加载的模块名前缀

    Returns:
        构造好的工具实例(尚未注册;注册交给 ToolRegistry.register)

    Raises:
        ValueError: 未知工具类型,或 custom 工具模块不在受信任命名空间
    """
    tool_type = tool_spec.get("type")
    config = tool_spec.get("config", {})

    if tool_type == "custom":
        # 只允许从受信任命名空间加载:module 必须匹配白名单前缀,
        # 否则任何可导入模块里的类都能被实例化,等于开放代码注入
        module_path = config.get("module")
        class_name = config.get("class")
        if not module_path or not class_name:
            raise ValueError("Custom tool spec requires 'module' and 'class'")
        if not str(module_path).startswith(allowed_prefixes):
            raise ValueError("Custom tool module is not in the trusted namespace")

        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        return tool_class(**config.get("args", {}))

    # 合并全局注册表与调用方临时提供的 builders(后者优先)
    merged = dict(_BUILDERS)
    if builders:
        merged.update(builders)
    if tool_type in merged:
        return merged[tool_type](config)

    raise ValueError(f"Unknown tool type: {tool_type}")


def build_and_register(tool_spec: Dict[str, Any], registry: ToolRegistry, **kwargs: Any) -> Tool:
    """构造并注册一步完成(对应文档图 5-1 的完整流程)

    新工具 -> build_tool -> 工具实例 -> register -> 注册中心(缓存 Schema)
    """
    tool = build_tool(tool_spec, **kwargs)
    registry.register(tool)
    return tool
