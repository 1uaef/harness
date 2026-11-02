# Author : atg
# Time   :  2026/11/1 11:02

"""纠正策略四:上下文重置(文档 4.5.4)

其他纠正手段(反思/回滚/约束)都失效时——Agent 深陷循环或
漂移太远——与其继续在污染的上下文里修修补补,不如"重启":
清空累积的失败尝试与中间推理,只保留关键信息从头再来。
Anthropic 在 Harness 设计实践中也采用这一模式。

执行流程(文档图 4-6):
    检测到循环/深度漂移
      -> 1. 提取并持久化关键信息(目标/已验证进展/学到的规则/失败模式)
      -> 2. 清空上下文窗口
      -> 3. 从检查点重新加载(注入进展摘要,规则写进系统提示)
      -> 4. 从干净状态继续执行
"""

from typing import List, Optional, Tuple

from runtime.message_types import AppState, Message, MessageRole, TextBlock

from .goal import Goal
from .checkpoint import CheckpointManager


class ContextResetManager:
    """上下文重置管理器"""

    def __init__(
        self,
        checkpoint_store: CheckpointManager,
        max_iterations_before_reset: int = 10,
        max_context_chars: int = 100_000,
        repeat_window: int = 5,
    ):
        self.checkpoint_store = checkpoint_store  # 与检查点策略共享的存储层
        self.max_iterations = max_iterations_before_reset  # 轮数保险丝(可选启用)
        self.max_context_chars = max_context_chars  # 上下文字符上限
        self.repeat_window = repeat_window  # 循环检测的行动窗口
        self.reset_count = 0  # 已执行的重置次数(可观测性)

    def should_reset(self, state: AppState, turn_count: Optional[int] = None) -> bool:
        """判断是否需要重置上下文

        触发条件(任一满足):
          1. 循环:最近窗口内的行动几乎完全重复(只有 0-1 种不同行动);
          2. 膨胀:消息历史文本超过 max_context_chars;
          3. 保险丝:调用方传入的 turn_count 达到 max_iterations
             (其他纠正手段反复无效时的最后防线)。
        """
        # 条件 1:重复行动 -> 陷入循环
        recent_actions = self._get_recent_actions(state, window=self.repeat_window)
        # 至少积累 3 个行动才谈得上"重复打转":
        # 刚开始工作的会话只有 1 种行动是正常的,不能误判为循环
        if len(recent_actions) >= 3 and len(set(recent_actions)) < 2:
            return True

        # 条件 2:上下文过度膨胀
        total_chars = sum(len(msg.get_text()) for msg in state.messages)
        if total_chars > self.max_context_chars:
            return True

        # 条件 3:轮数保险丝
        if turn_count is not None and turn_count >= self.max_iterations:
            return True

        return False

    def extract_and_save_checkpoint(self, state: AppState, goal: Goal) -> str:
        """提取关键信息并保存检查点,返回检查点 ID"""
        checkpoint = {
            "original_goal": goal.original_goal,
            "verified_progress": self._extract_verified_progress(state),
            "learned_constraints": self._extract_learned_rules(state),
            "failure_patterns": self._analyze_failure_patterns(state),
        }
        return self.checkpoint_store.save_snapshot(checkpoint)

    def reset_context(self, state: AppState, checkpoint_id: str) -> Tuple[AppState, str]:
        """重置上下文并从检查点恢复,返回(新状态, 系统提示)

        新状态只带一条"已验证进展"消息;学到的规则和失败模式
        写进系统提示,让模型带着经验从干净状态重新开始。
        """
        checkpoint = self.checkpoint_store.load_snapshot(checkpoint_id)

        # 创建新的干净状态(session 换新号,轮次从 0 重新计数)
        new_state = AppState(session_id=f"{state.session_id}_reset{self.reset_count + 1}")

        # 注入已验证的进展,作为继续执行的起点
        progress_msg = Message.assistant(
            [TextBlock(text=f"Previous verified progress:\n{checkpoint['verified_progress']}")]
        )
        new_state.add_message(progress_msg)

        # 构造系统提示,包含学到的规则与失败模式
        system_prompt = self._build_reset_system_prompt(checkpoint)

        self.reset_count += 1
        return new_state, system_prompt

    # ---- 关键信息提取 ----

    @staticmethod
    def _extract_verified_progress(state: AppState) -> str:
        """提取已验证完成的部分:最后一个成功返回的工具结果"""
        for msg in reversed(state.messages):
            for result in msg.get_tool_results():
                if not result.is_error:
                    return f"Successfully: {result.content[:500]}"
        return "No verified progress yet"

    @staticmethod
    def _extract_learned_rules(state: AppState) -> List[str]:
        """从失败和纠正过程中提取规则(扫最近 20 条消息中的'应避免'表述)"""
        rules = []
        markers = ("should not", "don't", "never", "不应该", "不要", "别再")
        for msg in state.messages[-20:]:
            text = msg.get_text()
            lower = text.lower()
            if any(marker in lower or marker in text for marker in markers):
                rules.append(text[:200])
        return rules

    @staticmethod
    def _analyze_failure_patterns(state: AppState) -> dict:
        """统计历史中的失败模式分布,供系统提示警示"""
        failure_types: dict = {}
        for msg in state.messages:
            for result in msg.get_tool_results():
                if result.is_error:
                    error_type = ContextResetManager._categorize_error(
                        result.error_type or result.content
                    )
                    failure_types[error_type] = failure_types.get(error_type, 0) + 1
        return failure_types

    @staticmethod
    def _categorize_error(error: str) -> str:
        """按关键词粗略归类错误"""
        for key in ("timeout", "permission", "not found", "rate limit"):
            if key in str(error).lower():
                return key
        return "other"

    @staticmethod
    def _build_reset_system_prompt(checkpoint: dict) -> str:
        """构造重置后的系统提示,包含学到的约束"""
        learned = "\n".join(f"- {rule}" for rule in checkpoint["learned_constraints"]) or "- (none)"
        prompt = f"""
You are solving: {checkpoint['original_goal']}

Previous verified progress:
{checkpoint['verified_progress']}

Learned constraints:
{learned}

Common failure patterns to avoid:
{checkpoint['failure_patterns']}

Continue from the verified progress and avoid previous mistakes.
"""
        return prompt.strip()

    def _get_recent_actions(self, state: AppState, window: int) -> List[str]:
        """获取最近的行动序列(工具名 + 参数签名,来自 goal.action_signature)"""
        from .goal import action_signature

        actions = []
        for msg in state.messages[-window * 2 :]:
            if msg.role is MessageRole.ASSISTANT and msg.has_tool_calls():
                for tool in msg.get_tool_calls():
                    actions.append(action_signature(tool))
        return actions
