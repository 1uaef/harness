# Author : atg
# Time   :  2026/11/1 11:02

"""纠正策略二:检查点与恢复(文档 4.5.3)

定期把会话状态存成检查点(持久化到磁盘 JSON);检测到漂移时
回滚到最近一个"正确"检查点,从那里重新开始,而不是从头再来。

检查点同时服务于上下文重置(策略四):重置前先落盘关键状态,
重置后从检查点恢复,二者共享同一个存储层。
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from runtime.message_types import AppState, Message

from .goal import Goal


@dataclass
class Checkpoint:
    """检查点:记录某个时刻的完整状态"""

    checkpoint_id: str
    turn_number: int
    session_id: str
    messages: List[Message] = field(default_factory=list)
    goal: Optional[Goal] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    drift_score: float = 0.0  # 保存时的漂移分数(用于事后分析漂移速度)


class CheckpointManager:
    """检查点管理"""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self, state: AppState, goal: Goal, drift_score: float = 0.0
    ) -> str:
        """保存检查点,返回检查点 ID"""
        checkpoint_id = f"ckpt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        filepath = self._filepath(checkpoint_id)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint_id": checkpoint_id,
                    "turn_number": state.current_turn,
                    "session_id": state.session_id,
                    "messages": [m.to_dict() for m in state.messages],
                    "goal": {
                        "original_goal": goal.original_goal,
                        "main_keywords": goal.main_keywords,
                        "metric": goal.metric,
                        "deadline": goal.deadline,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "drift_score": drift_score,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> Tuple[AppState, Goal]:
        """恢复检查点,返回(会话状态, 目标)"""
        filepath = self._filepath(checkpoint_id)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = [Message.from_dict(m) for m in data["messages"]]
        goal_data = data["goal"]
        goal = Goal(
            original_goal=goal_data["original_goal"],
            main_keywords=goal_data.get("main_keywords", []),
            metric=goal_data.get("metric"),
            deadline=goal_data.get("deadline"),
        )

        state = AppState(session_id=data.get("session_id", "sess_restored"))
        state.messages = messages
        state.current_turn = data["turn_number"]
        # Token 计数按恢复后的历史重算,保证与窗口管理口径一致
        state.token_count = sum(m.estimate_tokens() for m in messages)

        return state, goal

    def find_latest_before(self, turn_number: int) -> Optional[str]:
        """找到轮次早于 turn_number 的最近一个检查点(回滚用)

        回滚要回到"漂移发生之前"的状态,所以排除当前及之后的检查点;
        没有更早的检查点时返回 None,调用方退回纠正提示等其他策略。
        """
        candidates: Dict[int, str] = {}
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.checkpoint_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                turn = data["turn_number"]
                candidates[turn] = data["checkpoint_id"]
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # 单个损坏的检查点不影响整体恢复

        earlier = [t for t in candidates if t < turn_number]
        return candidates[max(earlier)] if earlier else None

    # ---- 通用快照存取(上下文重置管理器使用) ----

    def save_snapshot(self, data: Dict) -> str:
        """保存任意结构化快照(如上下文重置的关键信息提取),返回 ID"""
        snapshot_id = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        with open(self._filepath(snapshot_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Dict:
        """读取一个快照"""
        with open(self._filepath(snapshot_id), "r", encoding="utf-8") as f:
            return json.load(f)

    def _filepath(self, checkpoint_id: str) -> str:
        return os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
