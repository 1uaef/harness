# Author : atg
# Time   :  2026/11/1 11:02

"""漂移检测与纠正的共享类型

Goal 是"漂移"的参照物:没有明确的目标定义,漂移就无从谈起。
检测器拿 Goal 与最近行动比对,纠正策略拿 Goal 重新对齐方向。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Goal:
    """任务目标定义"""

    original_goal: str  # 原始目标的自然语言描述
    main_keywords: List[str] = field(default_factory=list)  # 从目标中提取的关键词(关键词检测器用)
    metric: Optional[str] = None  # 成功指标(可选)
    deadline: Optional[int] = None  # 轮数限制(可选)


def action_signature(tool_use) -> str:
    """把一次工具调用规范化为可比较的签名字符串

    例如 read_file(path=xxx) -> 'read_file({"path": "xxx"})'。
    名称 + 排序后的参数,两个"看起来一样"的调用签名一定相同,
    重复行动检测(局部最优/循环)都依赖它。
    """
    import json

    try:
        params = json.dumps(tool_use.input, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        params = str(tool_use.input)
    return f"{tool_use.name}({params})"
