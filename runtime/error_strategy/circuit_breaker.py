# Author : atg
# Time   :  2026/11/1 10:47

"""熔断器模式(文档 4.4.3)

对于可能频繁失败的外部调用(模型 API / 文件系统),熔断器防止
级联失败:连续失败达到阈值就"断开",后续请求快速失败,
不再打向已经故障的下游——既避免浪费资源,也给下游喘息时间。

状态转移(文档图 4-4):
    CLOSED(正常放行)
      └─ 连续失败 >= 阈值 ──> OPEN(拒绝所有请求)
                                └─ 冷却时间到 ──> HALF_OPEN(放一个探测请求)
                                    ├─ 探测成功 ──> CLOSED(恢复)
                                    └─ 探测失败 ──> OPEN(继续熔断)

对文档示例的三处加固:
    1. 抛 CircuitOpenError 而不是裸 Exception,调用方可精确捕获;
    2. 用 asyncio.Lock 保护状态检查与转移——多个并发协程共享
       同一个熔断器时,不加锁会有竞态(这正是发布-订阅一节
       说的"发布-订阅/共享可变状态需要额外同步机制");
    3. HALF_OPEN 严格放行"一个"探测请求,在途探测未返回前
       其余请求仍被拒绝,避免恢复期流量涌入打垮半死的下游。
"""

import asyncio
import time
from enum import Enum
from typing import Any, Optional

from .errors import CircuitOpenError


class CircuitBreakerState(str, Enum):
    """熔断器状态"""

    CLOSED = "closed"  # 正常,允许请求
    OPEN = "open"  # 失败过多,拒绝请求
    HALF_OPEN = "half_open"  # 尝试恢复,只允许一个探测请求


class CircuitBreaker:
    """熔断器模式"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "",
    ):
        self.failure_threshold = failure_threshold  # 连续失败多少次后熔断
        self.recovery_timeout = recovery_timeout  # OPEN 状态的冷却时长(秒)
        self.name = name or f"breaker_{id(self):x}"

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0  # 当前连续失败次数
        self.last_failure_time: Optional[float] = None  # 最近一次失败的时刻(monotonic)

        self._probe_in_flight = False  # HALF_OPEN 状态下是否已有探测在途
        self._lock = asyncio.Lock()  # 保护状态检查与转移

    async def call(self, coro_fn, *args, **kwargs) -> Any:
        """通过熔断器调用异步函数

        请求成功则原样返回结果;失败则原样抛出异常
        (熔断器只负责统计与熔断,不吞异常)。
        """
        # ---- 状态检查与准入判定(临界区) ----
        async with self._lock:
            if self.state is CircuitBreakerState.OPEN:
                if self._should_attempt_recovery():
                    # 冷却结束:转入半开,放行一个探测请求
                    self.state = CircuitBreakerState.HALF_OPEN
                    self._probe_in_flight = False
                else:
                    raise CircuitOpenError(
                        f"[{self.name}] 熔断器已打开,拒绝请求"
                        f"(约 {self.seconds_until_recovery():.0f}s 后进入半开)"
                    )
            if self.state is CircuitBreakerState.HALF_OPEN:
                if self._probe_in_flight:
                    # 已有探测在途:其余请求继续快速失败,防止恢复期流量涌入
                    raise CircuitOpenError(f"[{self.name}] 半开状态已有探测请求在途")
                self._probe_in_flight = True

        # ---- 实际调用(不持锁,不阻塞其他协程的状态查询) ----
        try:
            result = await coro_fn(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def _on_success(self) -> None:
        """成功后的处理:回到 CLOSED,清零失败计数"""
        async with self._lock:
            self.failure_count = 0
            self._probe_in_flight = False
            self.state = CircuitBreakerState.CLOSED

    async def _on_failure(self) -> None:
        """失败后的处理:累计失败;半开探测失败立即重新打开"""
        async with self._lock:
            self._probe_in_flight = False
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            # 半开状态下探测失败 -> 立即回到 OPEN(不用再凑满阈值);
            # CLOSED 状态下达到阈值 -> 打开熔断
            if self.state is CircuitBreakerState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

    def _should_attempt_recovery(self) -> bool:
        """判断冷却时间是否已过,可以进入半开状态"""
        if self.last_failure_time is None:
            return False
        # 用 monotonic 时钟:不受系统改时间影响,只关心"过去了多久"
        return time.monotonic() - self.last_failure_time > self.recovery_timeout

    def seconds_until_recovery(self) -> float:
        """距离可以尝试恢复还有多少秒(监控/日志用)"""
        if self.last_failure_time is None:
            return 0.0
        remaining = self.recovery_timeout - (time.monotonic() - self.last_failure_time)
        return max(0.0, remaining)
