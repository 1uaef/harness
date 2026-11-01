# Author : atg
# Time   :  2026/11/1 10:34

"""流式响应处理器(文档 4.3.3,OpenClaw 方案的异步化改写)

在流式响应中持续检测工具调用,核心边界(文档 4.3.1):
    - 文本增量:立即转发,用户立刻看到,不等任何东西;
    - 工具参数:input_json_delta 只能持续累积,必须等
      content_block_stop 到达、JSON 完整后才能调度执行
      ——半截 JSON 参数触发工具是流式处理最典型的错误。

本模块只做"流的解析与规范化",不执行工具:
执行由 StreamingToolExecutor 负责,调度由 QueryEngine 负责。
"""

import json
from typing import AsyncIterator, Dict, Optional

from runtime.message_types import ToolUseBlock

from .events import StreamEvent, StreamEventType


class StreamingResponseHandler:
    """处理 API 流式响应:把原始模型流规范化为统一事件流"""

    async def process(self, raw_stream: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
        """逐个处理流中的事件

        输入:模型客户端产出的原始流事件(CONTENT_BLOCK_START /
        TEXT_DELTA / TOOL_INPUT_DELTA / CONTENT_BLOCK_END / MESSAGE_END)。
        输出:规范化事件流——其中 TOOL_USE_END 携带参数完整的
        ToolUseBlock(metadata["tool_use"]),供下游调度执行。

        说明:文档示例用单个 current_tool_name/accumulated_json 变量,
        只能处理"同一时刻只有一个块在输出"的情况;这里按块的 index
        分桶,支持文本块与多个工具块交错输出。
        """
        # 每个内容块的累积状态:index -> {"type": ..., "tool_use_id": ..., "tool_name": ..., "json": ...}
        open_blocks: Dict[int, Dict] = {}

        async for event in raw_stream:
            if event.event_type == StreamEventType.CONTENT_BLOCK_START:
                block_type = event.metadata.get("block_type")
                index = event.metadata.get("index")
                if block_type == "tool_use":
                    # 工具调用开始:记下工具名和 id(content_block_stop 只带 index)
                    open_blocks[index] = {
                        "type": "tool_use",
                        "tool_use_id": event.metadata.get("tool_use_id", ""),
                        "tool_name": event.metadata.get("tool_name", ""),
                        "json": "",  # 参数 JSON 从零开始累积
                    }
                    # 转换为 TOOL_USE_START(此时参数还没到齐,不可执行)
                    yield StreamEvent.create(
                        StreamEventType.TOOL_USE_START,
                        tool_use_id=open_blocks[index]["tool_use_id"],
                        tool_name=open_blocks[index]["tool_name"],
                    )
                else:
                    # 文本块:直接放行
                    yield event

            elif event.event_type == StreamEventType.TEXT_DELTA:
                # 文本增量,直接转发给前端(低延迟路径,不等待任何东西)
                yield event

            elif event.event_type == StreamEventType.TOOL_INPUT_DELTA:
                # 工具参数增量:只累积,不转发、不执行
                index = event.metadata.get("index")
                if index in open_blocks:
                    open_blocks[index]["json"] += event.metadata.get("json_delta", "")

            elif event.event_type == StreamEventType.CONTENT_BLOCK_END:
                index = event.metadata.get("index")
                block = open_blocks.pop(index, None)
                if block and block["type"] == "tool_use":
                    # 工具块结束:参数 JSON 已完整,解析并交付完整调用
                    tool_use = self._build_tool_use(block)
                    if tool_use is None:
                        # JSON 解析失败(模型输出了残缺参数):
                        # 先上报 ERROR,再用空参数占位,保证 tool_use/tool_result
                        # 配对完整——工具收到空参数自己报错,错误结果会回传给模型修正
                        yield StreamEvent.create(
                            StreamEventType.ERROR,
                            stage="tool_input_parse",
                            tool_use_id=block["tool_use_id"],
                            raw_json=block["json"],
                        )
                        tool_use = ToolUseBlock(
                            id=block["tool_use_id"], name=block["tool_name"], input={}
                        )
                    yield StreamEvent.create(
                        StreamEventType.TOOL_USE_END,
                        tool_use_id=block["tool_use_id"],
                        tool_use=tool_use,
                    )
                # 内容块结束事件放行(文本块与工具块都发)
                yield event

            else:
                # MESSAGE_END 等其余事件原样放行
                yield event

    @staticmethod
    def _build_tool_use(block: Dict) -> Optional[ToolUseBlock]:
        """把累积完成的一个工具块状态解析成 ToolUseBlock

        参数 JSON 为空视为无参调用({});解析失败返回 None。
        """
        raw_json = block["json"].strip()
        try:
            tool_input = json.loads(raw_json) if raw_json else {}
        except json.JSONDecodeError:
            return None
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}  # 模型输出了非对象 JSON,包一层保住结构
        return ToolUseBlock(
            id=block["tool_use_id"],
            name=block["tool_name"],
            input=tool_input,
        )
