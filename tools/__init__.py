# Author : atg
# Time   :  2026/10/30 18:17

"""工具系统(文档 5.1 工具抽象接口设计)

模块划分:
    tool.py     泛型工具接口 Tool[InputType, OutputType] 与 ToolProgress
                (可选能力:权限检查 / 进度报告 / 流式输出)
    factory.py  build_tool() 工厂:按规范字典动态构造工具,
                builder 注册表扩展类型,custom 类型受信命名空间加载
    register.py ToolRegistry 注册中心:注册 / 发现 / Schema 缓存 /
                注销,工具生命周期管理

完整流程(文档图 5-1):
    新工具 --build_tool--> 工具实例 --register--> 注册中心(缓存 Schema)
    Agent --list_tools/get_tool_schema--> 决策 --get--> 实例调用

典型用法:
    from tools import Tool, ToolRegistry, build_tool, tool_builder

    @tool_builder("echo")
    def build_echo(config):
        return EchoTool(**config)

    registry = ToolRegistry()
    tool = build_and_register({"type": "echo", "config": {}}, registry)
"""

from .factory import ToolBuilder, build_and_register, build_tool, register_builder, tool_builder
from .register import ToolRegistry
from .tool import InputType, OutputType, Tool, ToolProgress

__all__ = [
    # 泛型接口
    "Tool",
    "ToolProgress",
    "InputType",
    "OutputType",
    # 注册中心
    "ToolRegistry",
    # 工厂
    "build_tool",
    "build_and_register",
    "tool_builder",
    "register_builder",
    "ToolBuilder",
]
