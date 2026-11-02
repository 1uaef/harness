# Author : atg
# Time   :  2026/11/1 11:02

"""长时任务的漂移检测与纠正(文档 4.5)

模块划分:
    goal.py              共享类型:Goal 目标定义、action_signature 行动签名
    keyword_detector.py  检测机制一:关键词相似度 + 范围蠕变 + 局部最优
    semantic_detector.py 检测机制二:向量余弦相似度(需嵌入模型)
    reflection.py        纠正策略一:强制反思步骤(定期自评)
    checkpoint.py        纠正策略二:检查点与恢复(回滚到正确状态)
    constraint.py        纠正策略三:约束验证(硬边界,越界即停)
    context_reset.py     纠正策略四:上下文重置(其他手段失效时的"重启")

组合方式(文档图 4-5 的流程):
    每轮用检测器算漂移分数 -> 超阈值按程度选纠正策略:
    早期漂移用反思;有检查点用回滚;硬约束被违反直接停;
    深陷循环/深度漂移用上下文重置。

典型用法:
    from runtime.drift_correct import (
        Goal, DriftDetector, ReflectionStep, CheckpointManager,
        default_validator, ContextResetManager,
    )

    goal = Goal(original_goal="读取报告并总结", main_keywords=["read", "report"])
    detector = DriftDetector()
    is_drifted, score, reason = detector.detect_drift(goal, app_state)
"""

from .checkpoint import Checkpoint, CheckpointManager
from .constraint import ConstraintValidator, default_validator
from .context_reset import ContextResetManager
from .goal import Goal, action_signature
from .keyword_detector import DriftDetector
from .reflection import ReflectionStep, analyze_reflection
from .semantic_detector import (
    EmbeddingModel,
    HashingEmbedder,
    SemanticDriftDetector,
)

__all__ = [
    # 共享类型
    "Goal",
    "action_signature",
    # 检测机制
    "DriftDetector",
    "SemanticDriftDetector",
    "EmbeddingModel",
    "HashingEmbedder",
    # 纠正策略
    "ReflectionStep",
    "analyze_reflection",
    "Checkpoint",
    "CheckpointManager",
    "ConstraintValidator",
    "default_validator",
    "ContextResetManager",
]
