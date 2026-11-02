# Author : atg
# Time   :  2026/11/1 11:02

"""漂移检测机制一:目标与当前行动的关键词相似度检查(文档 4.5.2)

思路:把目标提炼成关键词,检查最近 N 轮的工具调用与关键词的
匹配程度——匹配率低于阈值即判定漂移。另叠加两个启发式检测:
    - 范围蠕变:工具调用总数超过合理上限(成本失控的信号);
    - 局部最优:小窗口内重复执行相同调用(原地打转的信号)。

优点是零依赖、可解释(reason 直接告诉你是哪一项触发);
局限是关键词覆盖不全时会误报,语义层面的漂移要靠语义检测器。
"""

from typing import List, Tuple

from runtime.message_types import AppState, MessageRole

from .goal import Goal, action_signature


class DriftDetector:
    """漂移检测器(关键词相似度 + 范围蠕变 + 局部最优)"""

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        window_size: int = 5,
        max_tool_calls: int = 50,
        repeat_threshold: int = 2,
    ):
        self.similarity_threshold = similarity_threshold  # 匹配率低于该值判定漂移
        self.window_size = window_size  # 检查最近 N 轮的行动
        self.max_tool_calls = max_tool_calls  # 全程工具调用上限(范围蠕变)
        self.repeat_threshold = repeat_threshold  # 窗口内同一调用重复上限(局部最优)

    def detect_drift(self, goal: Goal, state: AppState) -> Tuple[bool, float, str]:
        """检测是否发生了漂移

        返回: (is_drifted, drift_score, reason)
        """
        recent_actions = self._get_recent_actions(state)

        # 主指标:最近行动与目标关键词的匹配率,太低说明在做无关的事
        drift_score = self._compute_drift_score(goal, recent_actions)

        if recent_actions and drift_score < self.similarity_threshold:
            reason = f"Low similarity with goal (score={drift_score:.2f})"
            return True, drift_score, reason

        # 辅助指标一:范围蠕变
        if self._detect_scope_expansion(state):
            return True, drift_score, "Scope expansion detected"

        # 辅助指标二:局部最优(原地打转)
        if self._detect_local_optimum(state):
            return True, drift_score, "Stuck in local optimum"

        return False, drift_score, "No drift detected"

    def _get_recent_actions(self, state: AppState) -> List[str]:
        """提取最近 window_size 轮(约 window_size*2 条消息)的工具调用"""
        recent = []
        for msg in state.messages[-self.window_size * 2 :]:
            if msg.role is MessageRole.ASSISTANT:
                for tool_use in msg.get_tool_calls():
                    # 动作串 = 工具名 + 参数名,供关键词匹配
                    recent.append(f"{tool_use.name}({', '.join(tool_use.input.keys())})")
        return recent

    def _compute_drift_score(self, goal: Goal, actions: List[str]) -> float:
        """计算动作与目标的相关性:匹配了至少一个关键词的动作占比

        说明:文档示例按(关键词×动作)配对计数再除以动作数,
        多关键词时会得到 >1 的"相似度";这里改为按动作归一化,
        保证分数落在 [0, 1]。
        """
        if not actions:
            return 0.0

        keywords = [kw.lower() for kw in goal.main_keywords]
        matched_actions = sum(
            1 for action in actions if any(kw in action.lower() for kw in keywords)
        )
        return matched_actions / len(actions)

    def _detect_scope_expansion(self, state: AppState) -> bool:
        """检测范围蠕变:全程工具调用数超过合理上限

        启发式:大多数任务不需要超过 max_tool_calls 次工具调用,
        超过即视为范围在不断扩大(文档默认 50)。
        """
        tool_count = sum(
            len(msg.get_tool_calls())
            for msg in state.messages
            if msg.role is MessageRole.ASSISTANT
        )
        return tool_count > self.max_tool_calls

    def _detect_local_optimum(self, state: AppState) -> bool:
        """检测是否陷入局部最优:窗口内同一调用(名称+参数)反复出现

        说明:文档示例用 input.get('path') 做签名,只适配文件类工具;
        这里改用 action_signature(名称+全部排序参数),对任何工具通用。
        """
        action_counts = {}
        for msg in state.messages[-self.window_size * 2 :]:
            if msg.role is MessageRole.ASSISTANT:
                for tool_use in msg.get_tool_calls():
                    key = action_signature(tool_use)
                    action_counts[key] = action_counts.get(key, 0) + 1

        # 窗口内有同一调用被反复执行超过 repeat_threshold 次 -> 原地打转
        return max(action_counts.values()) > self.repeat_threshold if action_counts else False
