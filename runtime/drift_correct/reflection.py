# Author : atg
# Time   :  2026/11/1 11:02

"""纠正策略一:强制反思步骤(文档 4.5.3,OpenClaw 方案)

每隔 N 轮在智能体循环中插入一次"反思点":把原始目标和最近
进度摆到模型面前,要求它自评是否仍在轨道上;反思判定偏离时,
追加纠正提示要求重新规划。

这是成本最低的纠正手段——不回滚、不清上下文,只是"问一问"。
适合漂移早期;深度漂移要靠检查点回滚或上下文重置。
"""

from typing import List

from runtime.message_types import AppState, Message, MessageRole

from .goal import Goal

# 反思回答中的"偏离"信号词(小写匹配)
OFF_TRACK_KEYWORDS = ["deviated", "off track", "wrong", "mistake", "偏离", "错误"]


def analyze_reflection(reflection_response: Message) -> bool:
    """分析反思结果:回答里出现偏离信号词即判定'已偏离'(返回 False)"""
    text = reflection_response.get_text().lower()
    return not any(kw in text for kw in OFF_TRACK_KEYWORDS)


class ReflectionStep:
    """强制反思步骤"""

    def __init__(self, reflection_interval: int = 5):
        # 每 reflection_interval 轮执行一次反思
        self.reflection_interval = reflection_interval

    def should_reflect(self, turn_count: int) -> bool:
        """判断当前轮次是否应该进行反思"""
        return turn_count > 0 and turn_count % self.reflection_interval == 0

    def build_reflection_prompt(self, goal: Goal, state: AppState) -> Message:
        """构造反思提示(以 user 消息形式注入对话)"""
        reflection_prompt = f"""
### Reflection Point

Original goal: {goal.original_goal}

Current progress (recent turns):
{self._summarize_recent_progress(state)}

Please answer:
1. Are we still on track to achieve the original goal?
2. Have we deviated from the original objective?
3. Should we adjust our approach?
4. What's the next best action?

Be concise and critical.
"""
        return Message.user(reflection_prompt)

    @staticmethod
    def analyze_reflection(reflection_response: Message) -> bool:
        """分析反思结果(委托给模块级 analyze_reflection)"""
        return analyze_reflection(reflection_response)

    @staticmethod
    def _summarize_recent_progress(state: AppState, max_messages: int = 6, max_chars: int = 200) -> str:
        """摘要最近的进度:最近几条 assistant 消息的文本截断拼接"""
        summary: List[str] = []
        recent = [m for m in state.messages[-max_messages:] if m.role is MessageRole.ASSISTANT]
        for msg in recent:
            text = msg.get_text()[:max_chars]  # 限制长度,反思提示本身不能太长
            summary.append(f"- {text}")
        return "\n".join(summary) if summary else "(no assistant output yet)"
