# Author : atg
# Time   :  2026/11/1 10:47

"""上下文压缩器(文档 4.4.1 第 4 类:上下文溢出的应对)

上下文溢出影响整个会话,恢复策略按信息损失从小到大:
    策略 1:对过长的 assistant 长响应做摘要(损失最小);
    策略 2:摘要后仍超预算,从最早的消息开始丢弃。

与 MessageWindow 的分工:窗口管理器是"每轮发送前的常规裁剪"
(确定性截断,不做摘要);压缩器是"溢出之后的恢复手段"
(可引入摘要模型,尽量保住信息)。前者防患未然,后者亡羊补牢。

Token 估算直接复用 Message.estimate_tokens(),全项目保持同一把尺子。
"""

from typing import Callable, List, Optional

from runtime.message_types import Message, TextBlock


class ContextCompressor:
    """上下文压缩器"""

    def __init__(self, max_summary_length: int = 500):
        self.max_summary_length = max_summary_length  # 摘要的最大长度(字符)

    def compress_message_history(
        self,
        messages: List[Message],
        target_tokens: int,
        summarizer: Optional[Callable[..., str]] = None,
    ) -> List[Message]:
        """压缩消息历史以满足 Token 预算

        summarizer:可选的摘要函数,签名为 summarizer(text, max_length=...)
        (通常接一个摘要模型)。不提供则直接走丢弃策略。

        返回新列表,不修改传入的历史(与 MessageWindow 同一约定:
        AppState 持有的完整历史只增不减,压缩只发生在读取路径上)。
        """
        # 拷贝后再删减,不污染调用方的历史
        messages = list(messages)
        total_tokens = self._estimate_tokens(messages)

        if total_tokens <= target_tokens:
            return messages  # 无需压缩

        # ---- 策略 1:先对过长的 assistant 长响应做摘要(信息损失最小) ----
        if summarizer:
            summarized: List[Message] = []
            for message in messages:
                if message.role.value == "assistant":
                    original_text = message.get_text()
                    if len(original_text) > self.max_summary_length:
                        message = self._summarize(message, original_text, summarizer)
                summarized.append(message)
            messages = summarized
            total_tokens = self._estimate_tokens(messages)

        # ---- 策略 2:摘要后仍超出预算,从最早的消息开始移除 ----
        while messages and total_tokens > target_tokens:
            removed = messages.pop(0)
            total_tokens -= removed.estimate_tokens()

        return messages

    def _summarize(self, message: Message, original_text: str, summarizer: Callable[..., str]) -> Message:
        """对单条消息做摘要;摘要函数本身失败时保留原文(压缩不能引入新故障)"""
        try:
            summary = summarizer(original_text, max_length=self.max_summary_length)
            # 摘要替换原内容,但保留 message_id/time,历史链条不断
            return Message(
                role=message.role,
                content=[TextBlock(text=summary)],
                timestamp=message.timestamp,
                message_id=message.message_id,
            )
        except Exception:
            return message

    @staticmethod
    def _estimate_tokens(messages: List[Message]) -> int:
        """估算一批消息的 Token 总量(复用 Message 自己的启发式估算)"""
        return sum(m.estimate_tokens() for m in messages)
