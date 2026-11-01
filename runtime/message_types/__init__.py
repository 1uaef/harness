# Author : atg
# Time   :  2026/11/1 10:20

"""消息类型系统与状态管理(文档 4.2)

模块划分:
    blocks.py   内容块类型:TextBlock / ToolUseBlock / ToolResultBlock /
                ThinkingBlock / ImageBlock,以及反序列化分派 parse_block
    message.py  消息对象 Message 与角色枚举 MessageRole
    state.py    发布-订阅模式的应用状态 AppState(Claude Code 方案)
    window.py   消息历史窗口管理 MessageWindow(条数/Token 截断、
                工具配对修复、系统上下文注入)

典型用法:
    from runtime.message_types import AppState, Message, MessageWindow

    state = AppState(session_id="sess_001")
    state.subscribe(lambda event, data: print(event, data))
    state.add_message(Message.user("你好"))
    state.execute_tool("tooluse_abc", "read_file", "文件内容")

    window = MessageWindow(max_messages=50, max_tokens=100000)
    context = window.insert_system_context(state.get_history(), ["你是一个助手"])
"""

from .blocks import (
    ContentBlock,
    ContentBlockType,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    parse_block,
)
from .message import Message, MessageRole
from .state import AppState, StateChangeCallback, StateChangeEvent
from .window import MessageWindow

__all__ = [
    # 内容块
    "ContentBlock",
    "ContentBlockType",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "ImageBlock",
    "parse_block",
    # 消息
    "Message",
    "MessageRole",
    # 状态管理(发布-订阅)
    "AppState",
    "StateChangeEvent",
    "StateChangeCallback",
    # 窗口管理
    "MessageWindow",
]
