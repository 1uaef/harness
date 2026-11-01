# Author : atg
# Time   :  2026/11/1 10:20

"""内容块(Content Block)类型定义

智能体循环中流转的不只是纯文本,一条消息的 content 由多种"内容块"组成:

    TextBlock        纯文本
    ToolUseBlock     工具调用请求(Agent 发起)
    ToolResultBlock  工具执行结果(回传给模型)
    ThinkingBlock    推理过程(extended thinking,可选)
    ImageBlock       图像内容(多模态,可选)

每个块都提供 to_dict() / from_dict(),支持消息历史序列化持久化后
再反序列化恢复,这是会话能够跨进程保存/恢复的基础。
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union


class ContentBlockType(str, Enum):
    """内容块类型枚举,与每个块的 type 字段对应"""

    TEXT = "text"                # 纯文本
    TOOL_USE = "tool_use"        # 工具调用请求
    TOOL_RESULT = "tool_result"  # 工具执行结果
    THINKING = "thinking"        # 推理过程(可选)
    IMAGE = "image"              # 图像内容(可选)


@dataclass
class TextBlock:
    """纯文本块:普通文字内容"""

    # type 由类自动维护,设为 keyword-only:
    # 避免 TextBlock("文本") 这类位置传参误绑到第一个字段 type 上
    type: str = field(default=ContentBlockType.TEXT.value, kw_only=True)
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {"type": self.type, "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        """从字典反序列化"""
        return cls(text=data.get("text", ""))


@dataclass
class ToolUseBlock:
    """工具调用块:Agent(模型)发起的工具调用请求

    id 是本次调用的唯一标识,用于和稍后回传的 ToolResultBlock.tool_use_id
    配对,保证一次调用恰好对应一个结果(窗口裁剪时也依赖这层配对关系)。
    """

    # 同 TextBlock:type 为 keyword-only,位置参数从 id 开始
    type: str = field(default=ContentBlockType.TOOL_USE.value, kw_only=True)
    id: str = ""  # 工具使用的唯一 ID,用于匹配结果
    name: str = ""  # 工具名称
    input: Dict[str, Any] = field(default_factory=dict)  # 工具参数,已解析为字典

    def __post_init__(self):
        # input 允许显式传 None,统一归一化为空字典
        if self.input is None:
            self.input = {}
        # 未指定 id 时自动生成,保证可配对、可追踪
        if not self.id:
            self.id = f"tooluse_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {"type": self.type, "id": self.id, "name": self.name, "input": self.input}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUseBlock":
        """从字典反序列化"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input") or {},
        )


@dataclass
class ToolResultBlock:
    """工具结果块:工具执行完成后的结果

    tool_use_id 必须指向某个 ToolUseBlock.id,模型据此知道
    这条结果对应哪一次调用。
    """

    # 同 TextBlock:type 为 keyword-only,位置参数从 tool_use_id 开始
    type: str = field(default=ContentBlockType.TOOL_RESULT.value, kw_only=True)
    tool_use_id: str = ""  # 关联的 ToolUseBlock.id
    content: str = ""  # 工具执行结果(文本形式)
    is_error: bool = False  # 是否为错误结果
    error_type: Optional[str] = None  # 错误类型(仅出错时存在)

    @classmethod
    def error(cls, tool_use_id: str, error_type: str, content: str = "") -> "ToolResultBlock":
        """工厂方法:构造一个错误结果块"""
        return cls(tool_use_id=tool_use_id, content=content, is_error=True, error_type=error_type)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResultBlock":
        """从字典反序列化"""
        return cls(
            tool_use_id=data.get("tool_use_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
            error_type=data.get("error_type"),
        )


@dataclass
class ThinkingBlock:
    """推理块:Agent 的内部思考过程(使用 extended thinking 时出现)"""

    # 同 TextBlock:type 为 keyword-only,位置参数从 thinking 开始
    type: str = field(default=ContentBlockType.THINKING.value, kw_only=True)
    thinking: str = ""  # 思考内容

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {"type": self.type, "thinking": self.thinking}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingBlock":
        """从字典反序列化"""
        return cls(thinking=data.get("thinking", ""))


@dataclass
class ImageBlock:
    """图像块:多模态输入,携带 base64 编码的图片数据"""

    # 同 TextBlock:type 为 keyword-only,位置参数从 media_type 开始
    type: str = field(default=ContentBlockType.IMAGE.value, kw_only=True)
    media_type: str = ""  # 图片 MIME 类型,如 image/png
    data: str = ""  # base64 编码的图片数据

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {"type": self.type, "media_type": self.media_type, "data": self.data}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageBlock":
        """从字典反序列化"""
        return cls(media_type=data.get("media_type", ""), data=data.get("data", ""))


# 一条消息的 content 就是这些块的组合
ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock, ImageBlock]

# type 字符串 -> 块类 的反序列化分派表
_BLOCK_TYPES = {
    ContentBlockType.TEXT.value: TextBlock,
    ContentBlockType.TOOL_USE.value: ToolUseBlock,
    ContentBlockType.TOOL_RESULT.value: ToolResultBlock,
    ContentBlockType.THINKING.value: ThinkingBlock,
    ContentBlockType.IMAGE.value: ImageBlock,
}


def parse_block(data: Dict[str, Any]) -> Optional[ContentBlock]:
    """根据字典中的 type 字段反序列化为对应的内容块

    未知类型返回 None 而不是抛异常:旧版本代码读到新版本写入的块类型时
    可以跳过,保证消息历史的向前兼容。
    """
    block_cls = _BLOCK_TYPES.get(data.get("type"))
    return block_cls.from_dict(data) if block_cls else None
