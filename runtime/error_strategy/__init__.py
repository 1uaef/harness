# Author : atg
# Time   :  2026/11/1 10:47

"""错误处理与故障恢复(文档 4.4)

模块划分:
    errors.py         四类错误的异常体系(工具/API/解析/上下文)+ 公共基类
    retry.py          重试策略:API 层指数退避 RetryPolicy、
                      工具层分类恢复 execute_tool_with_recovery
    validation.py     工具参数校验(输出解析错误的应对)
    compressor.py     上下文压缩器(上下文溢出的应对)
    observation.py    错误即观察:错误包装成对话内容反馈给 Agent
    circuit_breaker.py 熔断器:CLOSED/OPEN/HALF_OPEN 三态,防级联失败

典型用法:
    from runtime.error_strategy import (
        RetryPolicy, CircuitBreaker, execute_tool_with_recovery,
        execute_tool_validated, ErrorAsObservationHandler,
    )

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

    async def call_model():
        return await breaker.call(model_client.complete, messages)

    result = await RetryPolicy().execute_with_retry(call_model)
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .compressor import ContextCompressor
from .errors import (
    AgentError,
    CircuitOpenError,
    ContextOverflowError,
    ModelAPIError,
    ModelTimeoutError,
    OutputParsingError,
    RateLimitError,
    TokenBudgetExceededError,
    ToolExecutionError,
    ToolParameterValidationError,
    ToolPermissionError,
    ToolTimeoutError,
)
from .observation import ErrorAsObservationHandler
from .retry import RetryPolicy, execute_tool_with_recovery
from .validation import execute_tool_validated, raise_if_invalid, validate_tool_parameters

__all__ = [
    # 异常体系
    "AgentError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolPermissionError",
    "ModelAPIError",
    "RateLimitError",
    "ModelTimeoutError",
    "OutputParsingError",
    "ToolParameterValidationError",
    "ContextOverflowError",
    "TokenBudgetExceededError",
    "CircuitOpenError",
    # 重试与恢复
    "RetryPolicy",
    "execute_tool_with_recovery",
    # 校验
    "validate_tool_parameters",
    "execute_tool_validated",
    "raise_if_invalid",
    # 压缩
    "ContextCompressor",
    # 错误即观察
    "ErrorAsObservationHandler",
    # 熔断器
    "CircuitBreaker",
    "CircuitBreakerState",
]
