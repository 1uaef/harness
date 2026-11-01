# Author : atg
# Time   :  2026/11/1 10:34

"""流式处理与事件驱动架构(文档 4.3)

模块划分:
    events.py       流式事件类型 StreamEventType / StreamEvent(可 JSON 序列化)
    handler.py      StreamingResponseHandler:在响应流中解析工具调用——
                    文本增量立即转发,工具参数累积完整后才交付
    executor.py     StreamingToolExecutor:并发执行工具,信号量限流、
                    错误隔离、进度跟踪
    engine.py       QueryEngine:以事件流为输出的智能体循环,
                    以及模型流抽象 ModelStreamClient / 测试用 MockModelStream
    backpressure.py BackpressureQueue:有界队列,生产者被消费者"顶住",
                    防止事件堆积导致内存溢出

典型用法:
    from runtime.streaming import (
        QueryEngine, StreamingToolExecutor, MockModelStream, MessageWindow,
    )

    engine = QueryEngine(model=..., executor=..., window=..., system_prompt="...")
    async for event in engine.submit_message("帮我读文件"):
        print(event.to_json())  # 逐事件消费,可直接走 SSE/WebSocket
"""

from runtime.stream_executor.backpressure import BackpressureQueue
from runtime.stream_executor.engine import MockModelStream, ModelStreamClient, QueryEngine
from runtime.stream_executor.events import StreamEvent, StreamEventType
from runtime.stream_executor.executor import (
    PermissionDeniedError,
    StreamingToolExecutor,
    ToolNotFoundError,
    ToolProgress,
    ToolStatus,
)
from runtime.stream_executor.handler import StreamingResponseHandler

__all__ = [
    # 事件流
    "StreamEventType",
    "StreamEvent",
    # 响应流处理
    "StreamingResponseHandler",
    # 流式工具执行器
    "StreamingToolExecutor",
    "ToolProgress",
    "ToolStatus",
    "ToolNotFoundError",
    "PermissionDeniedError",
    # 查询引擎
    "QueryEngine",
    "ModelStreamClient",
    "MockModelStream",
    # 背压
    "BackpressureQueue",
]
