# Author : atg
# Time   :  2026/11/1 10:47

"""重试与恢复策略(文档 4.4.1 第 1、2 类错误的应对)

两层的重试策略:
    - RetryPolicy:模型 API 层——指数退避 + 速率限制感知
      (RateLimitError 按服务端告知的 retry_after 等待);
    - execute_tool_with_recovery:工具层——按错误子类分流:
      超时退避重试、权限错误立即放弃、普通工具错误快速重试。

重试的边界:重试只对"瞬时错误"有意义(超时、限流、偶发失败);
确定性错误(权限不足、参数非法)重试多少次结果都一样,只会拖慢循环。
"""

import asyncio
from typing import Any, Awaitable, Callable, Optional

from runtime.message_types import ToolResultBlock, ToolUseBlock
from tools.register import ToolRegistry

from .errors import ModelAPIError, RateLimitError, ToolExecutionError, ToolPermissionError, ToolTimeoutError


class RetryPolicy:
    """重试策略:模型 API 调用专用(指数退避 + 限流感知)"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries  # 最多尝试次数(含第一次,不只是"重试"次数)
        self.initial_backoff = initial_backoff  # 首次退避时长(秒)
        self.max_backoff = max_backoff  # 退避上限,防止指数爆炸
        self.exponential_base = exponential_base  # 指数底数

    async def execute_with_retry(self, coro_fn: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """执行异步操作,带重试

        策略按异常子类分流:
          - RateLimitError:等服务端告知的 retry_after(它最懂何时恢复);
          - ModelTimeoutError / 其他 ModelAPIError:指数退避
            initial_backoff * base^attempt,封顶 max_backoff;
        重试额度用尽后原样抛出最后一个异常,由调用方决定降级方式。
        """
        for attempt in range(self.max_retries):
            try:
                return await coro_fn(*args, **kwargs)

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(e.retry_after)
                    continue
                raise

            except ModelAPIError as e:
                # 超时与其他 API 错误:指数退避后重试
                if attempt < self.max_retries - 1:
                    backoff = min(
                        self.initial_backoff * (self.exponential_base ** attempt),
                        self.max_backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        # 循环正常耗尽(max_retries=0 等边界)也视为失败
        raise ModelAPIError("重试额度耗尽")


async def execute_tool_with_recovery(
    tool_use: ToolUseBlock,
    tool_registry: ToolRegistry,
    app_state=None,
    max_retries: int = 3,
    timeout_seconds: float = 10.0,
    initial_backoff: float = 1.0,
    backoff_base: float = 2.0,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> ToolResultBlock:
    """执行工具,带分类恢复机制(文档 4.4.1 的处理示例)

    按错误子类采取不同策略:
      - ToolPermissionError:不重试,立即返回错误块(重试必然同样失败);
      - ToolTimeoutError:指数退避后重试(下游可能只是暂时忙);
      - 其他 ToolExecutionError:立即重试(无退避,偶发失败下次就过);
      - 未预期异常:不重试,直接包装为错误块。

    所有失败最终都返回 is_error=True 的 ToolResultBlock 而不是抛异常
    ——错误要作为"观察"进入对话,让模型看到并自行调整(文档 4.4.2)。

    适配说明:文档示例假设 executor.execute 直接抛类型化异常;
    本项目 Tool.call 返回 ToolResult(success 标志)且超时由
    asyncio.wait_for 抛 TimeoutError,这里统一转换成类型化异常再分类。
    """
    # 前置检查:查不到工具 / 没有权限,连一次执行都不该发生
    tool = tool_registry.get(tool_use.name)
    if not tool:
        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=f"Tool '{tool_use.name}' not found",
            is_error=True,
            error_type="ToolNotFoundError",
        )
    if not tool.check_permissions(app_state):
        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=f"Permission denied for tool '{tool_use.name}'",
            is_error=True,
            error_type="ToolPermissionError",
        )

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            try:
                # wait_for 给工具执行加上硬超时,超时转成类型化 ToolTimeoutError
                raw = await asyncio.wait_for(tool.call(tool_use.input), timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise ToolTimeoutError(tool_use.name, f"执行超过 {timeout_seconds}s") from exc

            # 项目 Tool 约定:success=False 也用异常表达,交给下方统一分类
            if not raw.success:
                raise ToolExecutionError(tool_use.name, raw.content, error_type=raw.error_type)

            return ToolResultBlock(tool_use_id=tool_use.id, content=raw.content, is_error=False)

        except ToolPermissionError:
            # 权限错误:不重试,立即反馈
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Permission denied: {tool_use.name}",
                is_error=True,
                error_type="ToolPermissionError",
            )

        except ToolTimeoutError as e:
            # 超时:指数退避后重试;额度用尽则返回超时错误块
            if attempt < max_retries - 1:
                if on_retry:
                    on_retry(attempt, e)
                await asyncio.sleep(backoff)
                backoff *= backoff_base
                continue
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Tool timeout after {max_retries} attempts: {e.message}",
                is_error=True,
                error_type="ToolTimeoutError",
            )

        except ToolExecutionError as e:
            # 其他工具错误:立即重试(无退避);额度用尽则返回错误块
            if attempt < max_retries - 1:
                if on_retry:
                    on_retry(attempt, e)
                continue
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Tool execution failed: {e.message}",
                is_error=True,
                error_type=e.error_type,
            )

        except Exception as e:
            # 未预期的异常:不重试,原样包装
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Unexpected error: {e}",
                is_error=True,
                error_type=type(e).__name__,
            )

    # 理论不可达(循环内所有分支都有返回/继续),兜底保持类型完整
    return ToolResultBlock(
        tool_use_id=tool_use.id, content="unreachable", is_error=True, error_type="InternalError"
    )
