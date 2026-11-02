# Author : atg
# Time   :  2026/11/1 11:02

"""纠正策略三:约束验证(文档 4.5.3)

反思和回滚都是"事后纠正";约束验证是"事前防线"——每轮推理前
检查硬性界限(工具调用数、历史长度、执行时长),越界直接叫停,
从机制上保证范围蠕变和成本失控不可能发生。

约束是硬边界,违反即停,不做"纠正后再试";软性的方向纠偏
交给反思/检查点/重置等其他策略。
"""

from datetime import datetime, timezone
from typing import Callable, List, Tuple

from runtime.message_types import AppState, MessageRole


class ConstraintValidator:
    """约束验证器,确保智能体行为在预定界限内"""

    def __init__(self):
        self.constraints: List[dict] = []

    def add_constraint(
        self,
        name: str,
        check_fn: Callable[[AppState], bool],
        violation_msg: str,
    ) -> None:
        """添加约束条件(check_fn 返回 True 表示遵守)"""
        self.constraints.append(
            {
                "name": name,
                "check_fn": check_fn,
                "violation_msg": violation_msg,
            }
        )

    def validate(self, state: AppState) -> Tuple[bool, List[str]]:
        """验证所有约束,返回(是否全部通过, 违约消息列表)"""
        violations = []
        for constraint in self.constraints:
            try:
                ok = constraint["check_fn"](state)
            except Exception as exc:
                # 约束函数自身出错按违约处理:防线宁可误报也不能漏报;
                # 保留注册时的违约消息,附加错误详情便于排查
                ok = False
                msg = f"{constraint['violation_msg']} (check error: {exc})"
            else:
                msg = constraint["violation_msg"]
            if not ok:
                violations.append(msg)

        return len(violations) == 0, violations


def count_tool_calls(state: AppState) -> int:
    """统计全程的 assistant 工具调用总数(多个约束共用)"""
    return sum(
        len(msg.get_tool_calls()) for msg in state.messages if msg.role is MessageRole.ASSISTANT
    )


def total_text_chars(state: AppState) -> int:
    """统计消息历史的文本总字符数"""
    return sum(len(msg.get_text()) for msg in state.messages)


def default_validator(
    max_tool_calls: int = 50,
    max_chars: int = 1_000_000,
    max_seconds: float = 3600,
) -> ConstraintValidator:
    """构造文档示例的三条默认约束:
    工具调用数上限 / 消息历史字符上限 / 执行时长上限
    """
    validator = ConstraintValidator()

    validator.add_constraint(
        name="max_tool_calls",
        check_fn=lambda state: count_tool_calls(state) <= max_tool_calls,
        violation_msg=f"Exceeded maximum tool call limit ({max_tool_calls})",
    )

    validator.add_constraint(
        name="max_message_length",
        check_fn=lambda state: total_text_chars(state) <= max_chars,
        violation_msg=f"Message history exceeded {max_chars} characters",
    )

    validator.add_constraint(
        name="max_execution_time",
        check_fn=lambda state: (datetime.now(timezone.utc) - state.started_at).total_seconds() < max_seconds,
        violation_msg=f"Execution time exceeded {max_seconds} seconds",
    )

    return validator
