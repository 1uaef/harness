# Author : atg
# Time   :  2026/10/31 9:58

"""
工具注册中心(文档 5.1.5)

注册中心是所有工具的中枢,负责工具的注册、发现和管理:
    工具实例 --register--> 注册中心 --缓存--> Schema 缓存
    Agent 查询 --get/find--> 注册中心 --返回--> 工具实例 / 工具 Schema

Schema 在注册时一次性缓存,Agent 决策时查缓存即可,
避免每次都重新构造 Schema(文档图 5-1 的流程)。

类型说明:注解使用 tools.tool 的新版泛型 Tool;由于注册中心只依赖
name()/description()/input_schema() 三个方法,旧版 core.tool.Tool
的实例(接口一致)同样能正常注册和使用。
"""
from typing import Any, Dict, List, Optional

from tools.tool import Tool


class ToolRegistry:
    """工具注册中心,管理所有可用工具"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_cache: Dict[str, Dict[str, Any]] = {}  # Schema 缓存

    def register(self, tool: Tool):
        """注册工具(同名工具覆盖旧实例,Schema 同步刷新)"""
        name = tool.name()
        self.tools[name] = tool

        # 缓存工具的 Schema,供 Agent 决策时批量查询
        self.tool_cache[name] = {
            "name": name,
            "description": tool.description(),
            "input_schema": tool.input_schema(),
        }

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具实例"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具的 Schema(提供给模型做工具选择)"""
        return list(self.tool_cache.values())

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """获取单个工具的 Schema"""
        return self.tool_cache.get(name)

    def get_schema(self, name: str) -> Optional[Dict]:
        """获取单个工具的 Schema(早期版本的叫法,向后兼容别名)"""
        return self.get_tool_schema(name)

    def tool_exists(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self.tools

    def unregister(self, name: str) -> bool:
        """注销工具,同时清理 Schema 缓存

        Returns:
            是否确实注销了一个工具(不存在时返回 False,不抛错)
        """
        if name in self.tools:
            del self.tools[name]
            del self.tool_cache[name]
            return True
        return False
