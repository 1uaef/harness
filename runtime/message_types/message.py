# Author : atg
# Time   :  2026/11/1 10:20

"""消息(Message)对象定义

Message 是消息历史的最小单元:
    role     谁说的(user / assistant / system)
    content  内容块列表(文本、工具调用、工具结果等,见 blocks.py)

同一条消息的 content 可以混合多种块,例如一条 assistant 消息
同时包含 ThinkingBlock + TextBlock + ToolUseBlock。
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .blocks import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    parse_block,
)


class MessageRole(str, Enum):
    """消息角色

    继承 str 是为了 message.role == "user" 这样的比较也能成立,
    序列化时也可以直接取 .value。
    """

    USER = "user"  # 用户输入(工具结果也以 user 角色回传,这是 API 的约定)
    ASSISTANT = "assistant"  # 智能体(模型)响应
    SYSTEM = "system"  # 系统上下文(窗口管理器注入,不属于对话本身)


@dataclass
class Message:
    """完整的消息对象"""

    role: MessageRole  # "user" / "assistant" / "system"
    content: List[ContentBlock]  # 内容块列表,可混合多种类型
    timestamp: Optional[datetime] = None  # 创建时间(UTC)
    message_id: str = ""  # 唯一标识符,用于日志追踪与去重

    def __post_init__(self):
        # role 允许直接传字符串("user"),统一转成枚举
        if not isinstance(self.role, MessageRole):
            self.role = MessageRole(self.role)
        # content 允许直接传字符串,包装成单个文本块,降低构造成本
        if isinstance(self.content, str):
            self.content = [TextBlock(text=self.content)]
        # 未指定时间则取当前 UTC 时间
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        # 未指定 id 则自动生成
        if not self.message_id:
            self.message_id = f"msg_{uuid.uuid4().hex[:12]}"

    # ---------- 工厂方法 ----------

    @classmethod
    def user(cls, text: str) -> "Message":
        """工厂方法:创建用户消息"""
        return cls(role=MessageRole.USER, content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, content: List[ContentBlock]) -> "Message":
        """工厂方法:创建助手消息(可混合多种内容块)"""
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def system(cls, text: str) -> "Message":
        """工厂方法:创建系统上下文消息(由窗口管理器注入)"""
        return cls(role=MessageRole.SYSTEM, content=[TextBlock(text=text)])

    # ---------- 内容查询 ----------

    def has_tool_calls(self) -> bool:
        """检查是否包含工具调用"""
        return any(isinstance(block, ToolUseBlock) for block in self.content)

    def get_tool_calls(self) -> List[ToolUseBlock]:
        """提取所有工具调用块"""
        return [block for block in self.content if isinstance(block, ToolUseBlock)]

    def get_tool_results(self) -> List[ToolResultBlock]:
        """提取所有工具结果块"""
        return [block for block in self.content if isinstance(block, ToolResultBlock)]

    def get_text(self) -> str:
        """提取所有文本块,拼接为单个字符串"""
        texts = [block.text for block in self.content if isinstance(block, TextBlock)]
        return "".join(texts)

    # ---------- Token 估算 ----------

    def estimate_tokens(self) -> int:
        """粗略估计消息占用的 Token 数

        启发式近似,仅供窗口管理做预算控制,不代表真实 tokenizer 的结果:
          - 文本 / 思考块:按空白分词数 * 1.3(经验系数)
          - 工具参数 / 结果:按 JSON 字符数 / 4(近似比例)
        """
        tokens = 0.0
        for block in self.content:
            if isinstance(block, TextBlock):
                tokens += len(block.text.split()) * 1.3
            elif isinstance(block, ThinkingBlock):
                tokens += len(block.thinking.split()) * 1.3
            elif isinstance(block, ToolUseBlock):
                tokens += len(json.dumps(block.input, ensure_ascii=False)) / 4
            elif isinstance(block, ToolResultBlock):
                tokens += len(block.content) / 4
        return int(tokens)

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典(content 中每个块分别序列化)"""
        return {
            "role": self.role.value,
            "content": [block.to_dict() for block in self.content],
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典反序列化,按 type 字段逐块还原"""
        content: List[ContentBlock] = []
        for block_data in data.get("content", []):
            block = parse_block(block_data)
            if block is not None:
                content.append(block)
            # 未知块类型直接跳过(向前兼容,见 parse_block 注释)

        return cls(
            role=MessageRole(data["role"]),
            content=content,
            timestamp=datetime.fromisoformat(data.get("timestamp", "")),
            message_id=data.get("message_id", ""),
        )
