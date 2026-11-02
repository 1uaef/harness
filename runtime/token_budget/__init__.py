# Author : atg
# Time   :  2026/11/1 11:15

"""Token 预算:自动压缩与历史片段化(文档 4.6)

模块划分:
    compactor.py  AutoCompactor 自动压缩——使用率超阈值触发:
                  先移除最早消息,不够再摘要过长响应,最后兜底丢弃;
                  另含统一的 Token 估算 estimate_messages_tokens
    snipper.py    HistorySnipper 历史片段化——按位置(首条+最近 N 条)
                  或按重要性评分,从历史中挑出值得保留的片段

两者都会拆散 tool_use/tool_result 配对,保留结果统一经过
message_types.repair_tool_pairs 修复(孤儿块会被 API 直接拒绝)。

典型用法:
    from runtime.token_budget import AutoCompactor, HistorySnipper, estimate_messages_tokens

    compactor = AutoCompactor(compression_threshold=0.8)
    used = estimate_messages_tokens(state.get_history())
    if compactor.should_compact(used, total_budget=200_000):
        state.messages = compactor.compact_messages(state.messages, target_token_count=100_000)
        # 注意:AppState.messages 为列表,可直接整体替换;历史在压缩后仍然配对完整

    snipper = HistorySnipper()
    trimmed = snipper.snip_by_importance(state.get_history(), target_count=30)
"""

from .compactor import AutoCompactor, estimate_messages_tokens
from .snipper import HistorySnipper

__all__ = [
    # 自动压缩
    "AutoCompactor",
    "estimate_messages_tokens",
    # 历史片段化
    "HistorySnipper",
]
