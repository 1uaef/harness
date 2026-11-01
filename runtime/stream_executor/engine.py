# Author : atg
# Time   :  2026/11/1 10:34

"""查询引擎:以事件流为输出的智能体循环(文档 4.3.4)

传统批处理循环返回"最终结果";这里的循环是一个异步生成器,
边执行边 yield 事件——用户在第一轮就能看到文本增量,
工具调用在参数完整后立即调度,整个过程构成文档图 4-4 的事件序列。

依赖的三个组件:
    ModelStreamClient      模型流式客户端(产出原始流事件)
    StreamingResponseHandler  原始流 -> 规范化事件流(解析出完整工具调用)
    StreamingToolExecutor  并发执行工具(带信号量限流)
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List

from runtime.message_types import (
    AppState,
    Message,
    MessageRole,
    MessageWindow,
    TextBlock,
)

from .events import StreamEvent, StreamEventType
from .executor import StreamingToolExecutor
from .handler import StreamingResponseHandler


class ModelStreamClient(ABC):
    """模型流式客户端抽象

    真实实现对接 API SDK(把 SDK 的回调/迭代器翻译成这里的
    StreamEvent 流);测试用 MockModelStream 回放脚本。
    """

    @abstractmethod
    def stream(self, messages: List[Message], system: str) -> AsyncIterator[StreamEvent]:
        """流式调用模型,逐个产出原始流事件"""
        raise NotImplementedError


class MockModelStream(ModelStreamClient):
    """脚本化的模拟模型流:按顺序回放预设回复,用于测试与演示"""

    def __init__(self, scripted_replies: List[List[StreamEvent]]):
        self._script = list(scripted_replies)
        self._cursor = 0

    @staticmethod
    def text_reply(text: str) -> List[StreamEvent]:
        """构造一条纯文本回复的原始流事件"""
        return [
            StreamEvent.create(StreamEventType.CONTENT_BLOCK_START, index=0, block_type="text"),
            StreamEvent.create(StreamEventType.TEXT_DELTA, index=0, text=text),
            StreamEvent.create(StreamEventType.CONTENT_BLOCK_END, index=0, block_type="text"),
            StreamEvent.create(StreamEventType.MESSAGE_END, stop_reason="end_turn"),
        ]

    @staticmethod
    def tool_reply(
        text: str, tool_use_id: str, tool_name: str, tool_input: Dict
    ) -> List[StreamEvent]:
        """构造一条"文本 + 工具调用"的原始流事件

        参数 JSON 切成小的增量片段,模拟 input_json_delta 逐片到达,
        验证处理器"累积到完整才能执行"的边界。
        """
        events = [
            StreamEvent.create(StreamEventType.CONTENT_BLOCK_START, index=0, block_type="text"),
            StreamEvent.create(StreamEventType.TEXT_DELTA, index=0, text=text),
            StreamEvent.create(StreamEventType.CONTENT_BLOCK_END, index=0, block_type="text"),
            StreamEvent.create(
                StreamEventType.CONTENT_BLOCK_START,
                index=1,
                block_type="tool_use",
                tool_use_id=tool_use_id,
                tool_name=tool_name,
            ),
        ]
        # 参数 JSON 分片输出,模拟流式传输
        payload = json.dumps(tool_input, ensure_ascii=False)
        for i in range(0, len(payload), 6):
            events.append(
                StreamEvent.create(
                    StreamEventType.TOOL_INPUT_DELTA,
                    index=1,
                    tool_use_id=tool_use_id,
                    json_delta=payload[i : i + 6],
                )
            )
        events.append(
            StreamEvent.create(
                StreamEventType.CONTENT_BLOCK_END,
                index=1,
                block_type="tool_use",
                tool_use_id=tool_use_id,
            )
        )
        events.append(StreamEvent.create(StreamEventType.MESSAGE_END, stop_reason="tool_use"))
        return events

    async def stream(self, messages: List[Message], system: str) -> AsyncIterator[StreamEvent]:
        """回放下一段脚本;脚本耗尽后返回固定文本,防止测试死循环"""
        if self._cursor < len(self._script):
            events = self._script[self._cursor]
            self._cursor += 1
        else:
            events = self.text_reply("(脚本已耗尽)")
        for event in events:
            await asyncio.sleep(0)  # 让出控制权,模拟真实流的异步节奏
            yield event


class QueryEngine:
    """生成事件流的查询引擎"""

    def __init__(
        self,
        model: ModelStreamClient,
        executor: StreamingToolExecutor,
        window: MessageWindow,
        system_prompt: str = "",
        max_turns: int = 10,
    ):
        self.model = model
        self.executor = executor
        self.handler = StreamingResponseHandler()
        self.window = window  # 复用消息窗口管理器做上下文构建
        self.system_prompt = system_prompt
        self.max_turns = max_turns  # 轮次上限:模型反复要求调工具时强制刹车

    async def submit_message(self, user_input: str) -> AsyncIterator[StreamEvent]:
        """异步生成器:处理一次用户输入,逐个 yield 事件"""
        try:
            yield StreamEvent.create(StreamEventType.AGENT_START, user_input_length=len(user_input))

            app_state = AppState(session_id=f"sess_{uuid.uuid4().hex[:8]}")
            app_state.add_message(Message.user(user_input))
            turn_number = 0

            while True:
                turn_number += 1
                if turn_number > self.max_turns:
                    # 防御:模型可能无限循环要求调工具,超过上限直接报错终止
                    yield StreamEvent.create(
                        StreamEventType.ERROR,
                        error_type="MaxTurnsExceeded",
                        error_message=f"超过最大轮次限制({self.max_turns})",
                    )
                    break

                yield StreamEvent.create(
                    StreamEventType.TURN_START,
                    turn_number=turn_number,
                    context_tokens=app_state.token_count,
                )

                # 构建上下文:系统上下文 + 按窗口预算裁剪后的历史
                messages = self.window.insert_system_context(
                    app_state.get_history(), [self.system_prompt]
                )

                # 流式调用模型,经 handler 规范化后逐个转发
                tool_uses: List = []
                text_parts: List[str] = []
                stop_reason = ""
                async for event in self.handler.process(self.model.stream(messages, self.system_prompt)):
                    yield event

                    # 旁路收集:文本增量 / 完整工具调用 / 结束原因
                    if event.event_type == StreamEventType.TEXT_DELTA:
                        text_parts.append(event.metadata.get("text", ""))
                    elif event.event_type == StreamEventType.TOOL_USE_END:
                        tool_use = event.metadata.get("tool_use")
                        if tool_use is not None:
                            tool_uses.append(tool_use)
                    elif event.event_type == StreamEventType.MESSAGE_END:
                        stop_reason = event.metadata.get("stop_reason", "")

                # 本轮 assistant 消息(文本 + 完整工具调用)写入状态
                content = []
                if text_parts:
                    content.append(TextBlock(text="".join(text_parts)))
                content.extend(tool_uses)
                if content:
                    app_state.add_message(Message(role=MessageRole.ASSISTANT, content=content))

                # 工具输入完整后执行;多个工具已由执行器并发,这里等全部结果进下一轮
                if tool_uses:
                    tool_results = await self.executor.execute_tools(tool_uses, app_state)

                    # 发出工具结果事件
                    for tool_use_id, result in tool_results.items():
                        yield StreamEvent.create(
                            StreamEventType.TOOL_RESULT,
                            tool_use_id=tool_use_id,
                            is_error=result.is_error,
                            content_length=len(result.content),
                        )

                    # 结果写入状态,下一轮推理可见(API 约定:tool_result 以 user 角色进入历史)
                    app_state.add_tool_results(tool_results)

                    yield StreamEvent.create(
                        StreamEventType.TURN_END, turn_number=turn_number, has_tool_calls=True
                    )
                else:
                    # 没有工具调用,循环结束
                    break

            yield StreamEvent.create(
                StreamEventType.AGENT_END,
                final_response=app_state.get_final_response(),
                turn_count=turn_number,
                message_count=len(app_state.messages),
                stop_reason=stop_reason,
            )

        except Exception as e:
            # 引擎级兜底:任何未预期异常都转为 ERROR 事件而不是砸断事件流
            yield StreamEvent.create(
                StreamEventType.ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
            )
