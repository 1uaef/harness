# Author : atg
# Time   :  2026/11/1 11:15

"""自动压缩(文档 4.6.2 策略 2)

Token 使用接近预算阈值时自动压缩历史,策略按信息损失从小到大:
    1. 移除最早的消息(compact_messages);
    2. 还不够再摘要过长的 assistant 长响应(compact_with_summarization);
    3. 摘要后仍超预算,最后再丢弃(兜底)。

Token 口径:全项目统一使用 Message.estimate_tokens() 启发式估算,
与 MessageWindow / ContextCompressor 同一把尺子,避免各模块
对"用了多少 Token"各说各话。
"""

from typing import Callable, List, Optional, Union

from runtime.message_types import Message, TextBlock, repair_tool_pairs


def estimate_messages_tokens(messages: List[Message]) -> int:
    """估算一批消息的 Token 总量(统一走 Message.estimate_tokens)"""
    return sum(m.estimate_tokens() for m in messages)


# 摘要函数约定: summarizer(text, max_length=...) -> str
Summarizer = Callable[..., str]


class AutoCompactor:
    """自动压缩器:当 Token 使用超过阈值时触发压缩"""

    def __init__(self, compression_threshold: float = 0.8, max_summary_length: int = 500):
        # 触发压缩的使用率阈值:0.8 表示用到总预算 80% 时触发
        self.compression_threshold = compression_threshold
        self.max_summary_length = max_summary_length  # 超过该字符数的 assistant 响应才做摘要
        self.compaction_count = 0  # 已执行的压缩次数(可观测性)

    def should_compact(self, used_tokens: int, total_budget: int) -> bool:
        """判断是否应该进行压缩

        used_tokens 用 estimate_messages_tokens(messages) 计算,
        total_budget 为该任务的 Token 总预算。
        """
        return used_tokens > total_budget * self.compression_threshold

    def compact_messages(
        self,
        messages: List[Message],
        target_token_count: int,
    ) -> List[Message]:
        """压缩消息历史以满足 Token 目标(策略 1:移除最早的消息)

        返回新列表,不修改传入的历史;配对被拆散时自动修复
        (裁剪可能留下孤儿 tool_result/tool_use,API 会直接拒绝)。
        """
        current = estimate_messages_tokens(messages)
        if current <= target_token_count:
            return list(messages)  # 无需压缩

        self.compaction_count += 1

        # 从最旧的消息开始丢弃,累计扣减(避免每轮全量重算的 O(n²))
        compacted = list(messages)
        total = current
        # 至少保留 1 条:单条超长消息不应把历史清空
        while len(compacted) > 1 and total > target_token_count:
            removed = compacted.pop(0)
            total -= removed.estimate_tokens()

        # 丢弃可能拆散工具调用配对,统一修复
        return repair_tool_pairs(compacted)

    def compact_with_summarization(
        self,
        messages: List[Message],
        target_token_count: int,
        summarizer: Summarizer,
    ) -> List[Message]:
        """使用摘要进行压缩:先移除,不够再摘要,最后兜底丢弃

        比纯移除多保留信息:长响应折叠成摘要,而不是整体丢弃。
        """
        # 第一步:先尝试移除消息
        compacted = self.compact_messages(messages, target_token_count)
        if estimate_messages_tokens(compacted) <= target_token_count:
            return compacted

        # 第二步:摘要压缩过长的 assistant 响应
        compacted = self._apply_summarization(compacted, summarizer)
        # 摘要可能拆散配对(重建了消息内容),统一修复
        compacted = repair_tool_pairs(compacted)
        if estimate_messages_tokens(compacted) <= target_token_count:
            return compacted

        # 第三步:兜底——摘要后仍超预算,从最早的消息开始丢弃
        return self.compact_messages(compacted, target_token_count)

    def _apply_summarization(self, messages: List[Message], summarizer: Summarizer) -> List[Message]:
        """应用摘要压缩:过长的 assistant 响应折叠为摘要

        摘要函数失败时保留原文——压缩是有损操作,不能引入新故障。
        """
        result: List[Message] = []
        for message in messages:
            text = message.get_text()
            if message.role.value == "assistant" and len(text) > self.max_summary_length:
                try:
                    summary = summarizer(text, max_length=int(len(text) * 0.5))
                    # 摘要替换内容,但保留 message_id/time,历史链条不断
                    message = Message(
                        role=message.role,
                        content=[TextBlock(text=summary)],
                        timestamp=message.timestamp,
                        message_id=message.message_id,
                    )
                except Exception:
                    pass  # 摘要失败,保留原文
            result.append(message)
        return result
