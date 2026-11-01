# Author : atg
# Time   :  2026/11/1 10:34

"""事件流架构(文档 4.3.4)

智能体循环的每一步都产生事件,形成一条可观测的事件流:
系统行为因此可观测、可预测、可调试——UI 渲染、日志、指标
统计都是这条流的订阅方,而不是散落在循环里的 print。

事件序列(对应文档图 4-4):
    agent_start
      └─ turn_start
           └─ content_block_start / text_delta / tool_use_start
              / tool_input_delta / tool_use_end / content_block_end
           └─ message_end
           └─ tool_result(有工具调用时)
      └─ turn_end(有工具调用时,回到 turn_start)
    agent_end / error
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


class StreamEventType(str, Enum):
    """流式事件类型

    命名为 StreamEventType 而不是文档中的 EventType:
    项目里 core/event.py 已有一个任务层级的 EventType,
    这里是流式响应层级的事件,二者职责不同。
    """

    # ---- 生命周期 ----
    AGENT_START = "agent_start"  # 一次提问的处理开始
    AGENT_END = "agent_end"  # 全部轮次结束,拿到最终回答
    TURN_START = "turn_start"  # 一轮推理开始
    TURN_END = "turn_end"  # 一轮(含工具执行)结束
    MESSAGE_END = "message_end"  # 模型单条响应流结束

    # ---- 内容块流(模型逐增量输出) ----
    CONTENT_BLOCK_START = "content_block_start"  # 内容块开始(text / tool_use)
    TEXT_DELTA = "text_delta"  # 文本增量,可直接转发给前端
    TOOL_USE_START = "tool_use_start"  # 工具调用块开始(此时参数还没到齐)
    TOOL_INPUT_DELTA = "tool_input_delta"  # 工具参数 JSON 的增量片段
    TOOL_USE_END = "tool_use_end"  # 工具调用块结束,参数完整,可以调度执行
    CONTENT_BLOCK_END = "content_block_end"  # 内容块结束
    TOOL_RESULT = "tool_result"  # 工具执行结果

    ERROR = "error"  # 处理过程出错


@dataclass
class StreamEvent:
    """流式事件

    事件本身只携带 event_type + metadata,不定义强类型字段:
    流式场景下事件种类多、字段各异,用 metadata 字典保持
    序列化/传输的统一性(to_json 直接可用于 SSE/WebSocket)。
    """

    event_type: StreamEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON(用于流式传输)

        metadata 里可能携带富对象(如 TOOL_USE_END 中的 ToolUseBlock),
        进程内消费拿到的是原对象;序列化时对它们自动降级调用 to_dict(),
        保证事件流可以直接走 SSE/WebSocket。
        """
        return json.dumps(
            {
                "event_type": self.event_type.value,
                "timestamp": self.timestamp.isoformat(),
                "metadata": self.metadata,
            },
            ensure_ascii=False,
            default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o),
        )

    @classmethod
    def create(cls, event_type: StreamEventType, **metadata: Any) -> "StreamEvent":
        """便捷构造:StreamEvent.create(StreamEventType.TEXT_DELTA, text="你好")"""
        return cls(event_type=event_type, metadata=metadata)

    def __repr__(self) -> str:
        return f"StreamEvent({self.event_type.value}, {self.metadata})"
