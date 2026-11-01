# Author : atg
# Time   :  2026/11/1 10:34

"""背压与流量控制(文档 4.3.5)

流式处理中,如果生产者(模型/工具)产出事件的速度快于消费者
(UI/下游订阅方)的处理速度,无界队列会一直膨胀直到内存溢出。

背压(Backpressure)的思路:队列设上限——
    - 队列未满:事件立即入队,生产者不停;
    - 队列已满:put 挂起,生产者被"顶住",速度自动对齐消费者。
这样不需要丢弃事件,也不需要生产者显式查询消费者状态,
流量控制由队列本身完成。
"""

import asyncio

from .events import StreamEvent


class BackpressureQueue:
    """带背压的事件队列"""

    def __init__(self, max_queue_size: int = 100, get_timeout: float = 30.0):
        # 有界队列是背压的关键:满了之后 put 会挂起而不是继续堆积
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.get_timeout = get_timeout  # 取事件超时,默认 30 秒

    async def put_event(self, event: StreamEvent) -> None:
        """发送事件,如果队列满则等待(生产者在此被限流)"""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            # 队列满,挂起等待消费者腾出空间——这就是"背压"传导给生产者的时刻
            await self.queue.put(event)

    async def get_event(self) -> StreamEvent:
        """接收事件,带超时

        超时防止消费者在"生产者已死/流已中断"时永久阻塞;
        超时以 asyncio.TimeoutError 抛出,由调用方决定重试或放弃。
        """
        return await asyncio.wait_for(self.queue.get(), timeout=self.get_timeout)

    def put_event_nowait(self, event: StreamEvent) -> bool:
        """同步快速通道:队列满时立即返回 False 而不是等待

        供"宁可丢弃也不等"的场景使用(如日志采样),
        返回 True 表示入队成功。
        """
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    def queue_size(self) -> int:
        """获取当前队列大小"""
        return self.queue.qsize()

    def is_full(self) -> bool:
        """队列是否已满(监控/告警用)"""
        return self.queue.qsize() >= self.queue.maxsize
