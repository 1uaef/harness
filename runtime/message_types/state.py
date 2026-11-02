# Author : atg
# Time   :  2026/11/1 10:20

"""状态管理:发布-订阅模式(Pub-Sub)

Claude Code 采用的状态管理方式(文档 4.2.2):
维护一个可变的 AppState(原地修改,内存开销低、更新 O(1)),
但每次变更都通过事件广播通知订阅者,把"状态本体"与
"UI 渲染 / 日志 / 持久化"等旁路逻辑解耦——状态改动不需要
知道谁在关心它,新订阅方接入也不用改状态代码。

对比不可变状态模式(每次变更生成新对象、保留完整历史链):
见文档 4.2.3 的对比表,发布-订阅以"需要额外同步机制"换取
低内存开销与实时反应能力。
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Union

from .blocks import ToolResultBlock
from .message import Message, MessageRole


class StateChangeEvent(str, Enum):
    """状态变化事件类型"""

    MESSAGE_ADDED = "message_added"  # 有新消息进入历史
    TOOL_EXECUTED = "tool_executed"  # 一次工具调用执行完成
    STATE_RESET = "state_reset"  # 状态被重置(新会话/清空)


# 订阅者回调签名: callback(event, data)
StateChangeCallback = Callable[[StateChangeEvent, Any], None]


@dataclass
class AppState:
    """可变的应用状态(发布-订阅模式的核心)

    状态本身是普通可变对象;所有修改都应走 add_message / execute_tool /
    reset 方法,方法内部在修改完成后 _notify() 广播事件给订阅者,
    保证"状态变化"与"事件通知"总是成对发生。
    """

    session_id: str  # 会话唯一标识
    messages: List[Message] = field(default_factory=list)  # 完整消息历史(只增不减,裁剪交给 MessageWindow)
    current_turn: int = 0  # 当前对话轮次
    token_count: int = 0  # 历史消息的累计估算 Token 数
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # 会话开始时间(约束验证/漂移检测用)
    _subscribers: List[StateChangeCallback] = field(default_factory=list)  # 订阅者列表
    # 保护订阅者列表的锁:发布-订阅模式的并发安全靠它兜底(对比表中的"同步机制")
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # ---------- 订阅管理 ----------

    def subscribe(self, callback: StateChangeCallback) -> StateChangeCallback:
        """订阅状态变化事件,返回回调本身以便链式取消订阅"""
        with self._lock:
            self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: StateChangeCallback) -> None:
        """取消订阅(未注册的回调静默忽略)"""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify(self, event: StateChangeEvent, data: Any) -> None:
        """通知所有订阅者

        单个订阅者抛异常只打印、不向上传播:发布方无法(也不应)保证
        每个订阅方处理成功,一个坏掉的订阅者(如 UI 渲染出错)不能
        拖垮主循环。这是发布-订阅解耦的另一面。
        """
        with self._lock:
            subscribers = list(self._subscribers)  # 快照,避免遍历中被增删订阅者
        for callback in subscribers:
            try:
                callback(event, data)
            except Exception as exc:
                print(f"[AppState] 订阅者回调出错: {exc}")

    # ---------- 状态变更(先改状态,后广播) ----------

    def add_message(self, message: Message) -> Message:
        """添加消息到历史并广播 MESSAGE_ADDED 事件"""
        self.messages.append(message)
        self.token_count += message.estimate_tokens()
        self.current_turn += 1
        self._notify(StateChangeEvent.MESSAGE_ADDED, message)
        return message

    def execute_tool(
        self,
        tool_use_id: str,
        tool_name: str,
        result: str,
        is_error: bool = False,
        error_type: str = None,
    ) -> Message:
        """记录一次工具执行,做两件事:

        1. 把结果包装成 ToolResultBlock,以 user 消息的形式追加进历史
           (API 约定:tool_result 以 user 角色回传,并通过 tool_use_id
           与之前的 tool_use 配对);
        2. 广播 TOOL_EXECUTED 事件,供 UI/日志等订阅方实时展示执行情况。
        """
        result_block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=result,
            is_error=is_error,
            error_type=error_type,
        )
        message = self.add_message(Message(role=MessageRole.USER, content=[result_block]))
        self._notify(
            StateChangeEvent.TOOL_EXECUTED,
            {
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "result": result,
                "is_error": is_error,
                "message_id": message.message_id,
            },
        )
        return message

    def add_tool_results(self, results: Dict[str, ToolResultBlock]) -> Message:
        """把一轮执行的全部工具结果作为一条 user 消息写入历史

        API 约定:同一轮的多个 tool_result 块放在同一条 user 消息里,
        与之前 assistant 消息中的多个 tool_use 通过 id 一一配对。
        (QueryEngine 每轮工具执行完成后调用,供下一轮推理使用)
        """
        blocks = list(results.values())
        return self.add_message(Message(role=MessageRole.USER, content=blocks))

    def reset(self) -> None:
        """重置状态(清空历史、轮次、Token 计数)并广播 STATE_RESET"""
        self.messages = []
        self.current_turn = 0
        self.token_count = 0
        self._notify(StateChangeEvent.STATE_RESET, {"session_id": self.session_id})

    # ---------- 只读访问 ----------

    def get_history(self) -> List[Message]:
        """返回消息历史的浅拷贝,防止外部误改内部列表"""
        return list(self.messages)

    def last_message(self) -> Union[Message, None]:
        """最近一条消息,历史为空时返回 None"""
        return self.messages[-1] if self.messages else None

    def get_final_response(self) -> str:
        """最终回答:最后一条 assistant 消息的文本(没有则返回空串)"""
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT:
                return message.get_text()
        return ""
