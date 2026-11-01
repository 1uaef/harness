# Author : atg
# Time   :  2026/11/1 10:47

"""错误即观察(文档 4.4.2,OpenClaw 的设计哲学)

核心思想:错误不抛给上层中断循环,而是包装成对话内容
"反馈"给 Agent——

    - 工具错误 -> ToolResultBlock(is_error=True):模型能看到
      错误类型和消息,换一种方式重试(文件不存在?试试列目录);
    - API 错误 -> 特殊 assistant 消息:告知系统会自动重试;
    - 上下文溢出 -> 压缩历史后继续。

优点:智能体可以看到所有错误信息并学习处理;系统行为更透明;
智能体可以提出替代方案,而不是整个循环崩溃。
"""

from typing import Any, List

from runtime.message_types import Message, TextBlock, ToolResultBlock, ToolUseBlock

from .compressor import ContextCompressor
from .errors import ModelAPIError


class ErrorAsObservationHandler:
    """将错误作为观察反馈给 Agent"""

    def __init__(self, compressor: ContextCompressor = None):
        # 上下文溢出的恢复依赖压缩器,默认给一个
        self.compressor = compressor or ContextCompressor()

    def handle_tool_error(self, tool_use: ToolUseBlock, error: Any) -> ToolResultBlock:
        """处理工具错误:构造结构化的错误观察,回传给模型

        注意返回 ToolResultBlock 而不是抛异常——tool_use/tool_result
        的配对也因此保持完整,下一轮推理能正常进行。
        """
        error_message = f"[Tool Error]\nTool: {tool_use.name}\n"
        error_message += f"Error Type: {getattr(error, 'error_type', None) or type(error).__name__}\n"
        error_message += f"Error Message: {str(error)}\n"
        error_message += "Please analyze the error and try a different approach."

        return ToolResultBlock(
            tool_use_id=tool_use.id,
            content=error_message,
            is_error=True,
            error_type=type(error).__name__,
        )

    def handle_api_error(self, error: ModelAPIError) -> Message:
        """处理 API 错误:返回一条特殊的 assistant 消息告知用户/智能体

        这类错误发生在"模型说话"这一环,只能以消息形式表达;
        实际的重试由 RetryPolicy/熔断器在引擎层完成。
        """
        return Message.assistant(
            [
                TextBlock(
                    text=(
                        "An error occurred while processing your request:\n"
                        f"Error: {str(error)}\n"
                        "The system will retry the request automatically."
                    )
                )
            ]
        )

    def handle_context_overflow(self, messages: List[Message], token_budget: int) -> List[Message]:
        """处理上下文溢出:压缩历史来恢复

        目标压到预算的 50%,给后续轮次留出增长空间,
        避免每轮都贴着预算边缘反复触发压缩。
        """
        return self.compressor.compress_message_history(
            messages, target_tokens=int(token_budget * 0.5)
        )
