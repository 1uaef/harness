# Author : atg
# Time   :  2026/11/2 18:22

"""工具抽象接口:泛型设计(文档 5.1.2,Claude Code 风格)

Tool<Input, Output, Progress> 三泛型把工具的"输入类型"和"输出类型"
绑定到具体子类上——静态检查器(mypy/pyright)可以验证 call() 的
输入输出是否与声明一致;Progress 由下方具体的 ToolProgress 类承载
(简化实现,不再单设类型变量)。

⚠️ 类型系统权衡:Python 运行时是动态类型,泛型注解主要服务于
静态分析与可读性;运行时参数仍常以 Dict[str, Any] 形式流转,
由 input_schema() 声明的 JSON Schema 做运行时校验
(见 runtime/error_strategy/validation.py)。

可选能力(子类按需覆盖,默认不开):
    check_permissions   权限检查(默认放行)
    get_progress        进度报告(默认 None = 不支持)
    supports_streaming / stream_output  流式输出(默认不支持)

说明:core/tool.py 是项目早期简化版接口,二者方法签名一致;
基于旧接口的模块(runtime/*)无需改动即可继续工作。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

# 类型变量:分别绑定工具的输入与输出类型
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class ToolProgress:
    """工具执行进度"""

    step: int  # 当前步骤号
    total_steps: int  # 总步骤数
    current_status: str  # 当前状态描述
    estimated_time_remaining: Optional[float] = None  # 剩余时间(秒)

    def percentage(self) -> float:
        """进度百分比"""
        return (self.step / self.total_steps * 100) if self.total_steps > 0 else 0


class Tool(ABC, Generic[InputType, OutputType]):
    """通用工具抽象:Tool[InputType, OutputType]"""

    @abstractmethod
    async def call(self, input_data: InputType) -> OutputType:
        """执行工具

        Args:
            input_data: 工具输入,应已按 input_schema 验证过

        Returns:
            工具执行结果

        Raises:
            ToolExecutionError: 工具执行失败(见 runtime/error_strategy)
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """工具名称(注册中心以此为唯一键)"""
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """工具描述(提供给模型做工具选择)"""
        raise NotImplementedError

    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """输入的 JSON Schema"""
        raise NotImplementedError

    def check_permissions(self, context: Any) -> bool:
        """检查当前上下文是否有权限调用此工具

        Args:
            context: 执行上下文(包含用户、会话信息等)

        Returns:
            是否有权限(默认放行)
        """
        return True

    async def get_progress(self) -> Optional[ToolProgress]:
        """获取工具执行进度

        Returns:
            进度对象;不支持进度报告的工具返回 None
        """
        return None

    def supports_streaming(self) -> bool:
        """工具是否支持流式输出"""
        return False

    async def stream_output(self, input_data: InputType):
        """流式输出(可选能力)

        默认实现:不支持流式的工具把单次 call 的结果作为唯一的
        输出项产出;支持流式的子类应同时覆盖本方法,
        逐个 yield 产生的输出(声明 supports_streaming 为 True 却
        不覆盖,只会产不出任何结果——子类自己的责任)。

        Yields:
            OutputType: 逐个产生的输出
        """
        if not self.supports_streaming():
            output = await self.call(input_data)
            yield output
