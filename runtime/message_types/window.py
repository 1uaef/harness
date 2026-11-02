# Author : atg
# Time   :  2026/11/1 10:20

"""消息历史的窗口管理

LLM 的上下文窗口有限,消息历史不能无限增长地整体发给模型。
MessageWindow 负责在"发给模型之前"裁剪历史(文档 4.2.4):

    1. 按消息条数截断:只保留最近 N 条;
    2. 按 Token 预算截断:从最旧的消息开始丢弃,直到预算内;
    3. 修复工具调用配对:tool_use / tool_result 必须成对出现,
       截断可能把配对拆散,直接发给 API 会被拒绝;
    4. 注入系统上下文:系统消息放最前,且不占窗口预算。

注意分工:AppState 持有的完整历史只增不减(便于回溯/持久化),
窗口裁剪只发生在读取路径上,不修改 AppState。
"""

from dataclasses import replace
from typing import List, Union

from .blocks import TextBlock, ToolResultBlock, ToolUseBlock
from .message import Message, MessageRole


class MessageWindow:
    """消息窗口管理器"""

    def __init__(self, max_messages: int = 50, max_tokens: int = 100000):
        self.max_messages = max_messages  # 窗口内最多保留的消息条数
        self.max_tokens = max_tokens  # 窗口内消息的 Token 预算

    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """根据约束条件裁剪消息历史

        返回新列表,不修改传入的历史(AppState 里的完整历史保持原样)。
        """
        # 拷贝一份再截断,后续的切片/弹出都不影响调用方持有的列表
        messages = list(messages)

        # ---- 策略 1:按消息条数限制,只保留最近 max_messages 条 ----
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages :]

        # ---- 策略 2:按 Token 数限制,从最旧的消息开始丢弃 ----
        total_tokens = sum(m.estimate_tokens() for m in messages)
        # 至少保留 1 条:单条超长消息不应把窗口清空
        while len(messages) > 1 and total_tokens > self.max_tokens:
            removed = messages.pop(0)
            total_tokens -= removed.estimate_tokens()

        # ---- 策略 3:修复被截断破坏的工具调用配对 ----
        return repair_tool_pairs(messages)

    def insert_system_context(
        self,
        messages: List[Message],
        system_blocks: List[Union[str, TextBlock]],
    ) -> List[Message]:
        """在裁剪后的消息历史前插入系统上下文

        系统上下文不参与条数/Token 截断(不占窗口预算),
        因此无论对话多长,模型始终能看到完整的系统指令。
        """
        result: List[Message] = []
        for block in system_blocks:
            # 接受纯字符串或 TextBlock,统一包装成 system 角色消息
            text_block = block if isinstance(block, TextBlock) else TextBlock(text=block)
            result.append(Message(role=MessageRole.SYSTEM, content=[text_block]))
        # 追加按预算裁剪后的消息历史
        result.extend(self.filter_messages(messages))
        return result


def repair_tool_pairs(messages: List[Message]) -> List[Message]:
    """保证消息列表内的 tool_use 与 tool_result 成对出现(模块级公开函数)

    任何"选择性保留消息"的裁剪(窗口截断/自动压缩/历史片段化)都可能把
    配对拆散——留下找不到 tool_use 的孤儿 tool_result,或反之。
    带着"孤儿块"请求 API 会直接报错,所以裁剪后统一做双向清理:
      1. 删掉找不到对应 tool_use 的 tool_result;
      2. 删掉找不到对应 tool_result 的 tool_use。
    清理后 content 为空的消息整体移除。

    注意:消息对象与 AppState 共享,不能原地改 content,
    必须用 dataclasses.replace 生成新 Message,只替换裁剪后的副本。
    """
    # 第一遍:收集列表内所有 tool_use 的 id,清理孤儿 tool_result
    use_ids = {block.id for msg in messages for block in msg.get_tool_calls()}

    kept: List[Message] = []
    for msg in messages:
        if any(isinstance(b, ToolResultBlock) for b in msg.content):
            content = [
                b
                for b in msg.content
                if not isinstance(b, ToolResultBlock) or b.tool_use_id in use_ids
            ]
            if not content:
                continue  # 整条都是孤儿结果 -> 丢弃该消息
            if len(content) != len(msg.content):
                # 只在确实过滤掉块时才重建消息:配对完整时保留原对象,
                # 避免无谓的 replace 破坏对象身份(调用方可能按身份持有消息)
                msg = replace(msg, content=content)
        kept.append(msg)

    # 第二遍:收集留下的 tool_result 的 id,反向清理孤儿 tool_use
    result_ids = {
        b.tool_use_id
        for msg in kept
        for b in msg.content
        if isinstance(b, ToolResultBlock)
    }
    repaired: List[Message] = []
    for msg in kept:
        if msg.has_tool_calls():
            content = [
                b for b in msg.content if not isinstance(b, ToolUseBlock) or b.id in result_ids
            ]
            if not content:
                continue  # 整条都是孤儿调用 -> 丢弃该消息
            if len(content) != len(msg.content):
                msg = replace(msg, content=content)
        repaired.append(msg)

    return repaired
