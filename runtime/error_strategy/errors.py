# Author : atg
# Time   :  2026/11/1 10:47

"""错误类型体系(文档 4.4.1)

智能体循环中的错误按"影响范围 + 恢复策略"分为四类,
不同类别对应不同的应对方式,这是错误处理的第一个原则:

    类别            影响范围        恢复策略
    ─────────────────────────────────────────────────
    工具执行错误    当前工具调用    作为"观察"反馈给 Agent(可重试)
    API 调用错误    当前推理轮次    重试(指数退避)或降级
    输出解析错误    当前推理结果    参数校验 / 要求模型重新生成
    上下文溢出      整个会话        压缩历史 / 清理 / 新会话

所有异常都挂在 AgentError 基类下:调用方可以用一个
except AgentError 兜住全部已知错误,再按子类细分处理。
"""

from typing import List


class AgentError(Exception):
    """智能体运行时错误基类:统一兜底用"""

    def __init__(self, message: str, error_type: str = None):
        self.message = message
        # 默认用类名作为 error_type,便于写入 ToolResultBlock.error_type
        self.error_type = error_type or type(self).__name__
        super().__init__(message)


# ============ 1. 工具执行错误(影响:当前工具调用) ============

class ToolExecutionError(AgentError):
    """工具执行错误:调用失败 / 参数错误 / 工具崩溃

    retry_count 记录已重试次数,便于反馈给 Agent 时说明情况。
    """

    def __init__(self, tool_name: str, message: str, error_type: str = None, retry_count: int = 0):
        self.tool_name = tool_name
        self.retry_count = retry_count
        super().__init__(message, error_type=error_type)


class ToolTimeoutError(ToolExecutionError):
    """工具执行超时:可重试(指数退避)"""


class ToolPermissionError(ToolExecutionError):
    """工具权限不足:重试没有意义,不重试"""


# ============ 2. API 调用错误(影响:当前推理轮次) ============

class ModelAPIError(AgentError):
    """模型 API 错误:超时 / 限流 / 认证失败 / 服务中断"""


class RateLimitError(ModelAPIError):
    """速率限制:按服务端告知的 retry_after 等待后重试"""

    def __init__(self, retry_after: int = None):
        self.retry_after = retry_after or 60
        super().__init__(f"Rate limited, retry after {self.retry_after}s")


class ModelTimeoutError(ModelAPIError):
    """模型调用超时:指数退避重试"""


# ============ 3. 输出解析错误(影响:当前推理结果) ============

class OutputParsingError(AgentError):
    """输出解析错误:模型输出格式不符 / JSON 解析失败"""


class ToolParameterValidationError(OutputParsingError):
    """工具参数验证失败:携带逐条错误说明,反馈给模型重新生成"""

    def __init__(self, tool_name: str, errors: List[str]):
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(f"Tool '{tool_name}' parameter validation failed: {errors}")


# ============ 4. 上下文溢出(影响:整个会话) ============

class ContextOverflowError(AgentError):
    """上下文溢出:消息历史太长 / 输出超过 max_tokens"""


class TokenBudgetExceededError(ContextOverflowError):
    """Token 预算超支:需要压缩历史或开新会话"""

    def __init__(self, current_tokens: int, budget: int):
        self.current_tokens = current_tokens
        self.budget = budget
        super().__init__(f"Token budget exceeded: {current_tokens} > {budget}")


# ============ 熔断器错误 ============

class CircuitOpenError(AgentError):
    """熔断器打开:请求在到达目标之前就被拒绝

    这是熔断器的"快速失败"信号,调用方应走降级路径
    (备用模型 / 缓存结果 / 友好提示),而不是重试。
    """
