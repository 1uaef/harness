# Author : atg
# Time   :  2026/11/1 11:15

"""历史片段化(文档 4.6.2 策略 3):选择性保留消息

压缩是"把历史变短",片段化是"从历史里挑出值得保留的片段":
    - snip_history:保留首条消息(初始上下文)+ 最近 N 条(当前话题),
      中间整段跳过;
    - snip_by_importance:按重要性打分挑出 N 条,按原顺序拼接。

两种方式都以"条数"为预算,适合"消息条数可控、单条长度稳定"
的会话;按 Token 精确控制请配合 AutoCompactor 使用。

与压缩器一样,筛选会拆散 tool_use/tool_result 配对,
保留结果统一经过 repair_tool_pairs 修复。
"""

from typing import Callable, List, Optional

from runtime.message_types import Message, MessageRole, repair_tool_pairs

# 重要性评分函数约定: scorer(msg, index, total) -> float(0~1)
ImportanceScorer = Callable[[Message, int, int], float]


class HistorySnipper:
    """历史片段化:选择性保留消息"""

    def snip_history(self, messages: List[Message], target_count: int = 20) -> List[Message]:
        """按位置片段化:保留最重要的消息

        1. 保留最后 target_count 条消息(最近的对话);
        2. 保留第一条消息(初始上下文,通常是任务描述);
        3. 移除中间的消息。

        修正文档示例的一个 bug:文档先取最后 N 条再补首条,
        首条不在窗口内时会返回 target_count + 1 条,超出预算;
        这里为首条预留一个位置,总数严格不超过 target_count。
        """
        if len(messages) <= target_count:
            return list(messages)

        # 首条 + 最近 (target_count - 1) 条,总数恰好 target_count
        kept = messages[:1] + messages[-(target_count - 1) :]

        # 片段化会拆散工具调用配对(中间消息被跳过),统一修复
        return repair_tool_pairs(kept)

    def snip_by_importance(
        self,
        messages: List[Message],
        target_count: int = 20,
        importance_scorer: Optional[ImportanceScorer] = None,
    ) -> List[Message]:
        """基于重要性评分进行片段化

        importance_scorer:返回消息重要性分数(0~1)的函数;
        不提供时使用默认评分(越新越重要 + 工具调用重要 + 首条重要)。

        选取以"配对单元"为单位:tool_use 消息与其 tool_result 消息
        不可拆散——若各自独立评分,常常选中调用、漏选结果,配对修复
        又会把调用剥掉,高分消息反而被削弱。单元的分数取成员最高分,
        大小计入条数预算,贪心装填直到填满 target_count 条。

        选完后按原始顺序重新排列,保证对话时间线不被打乱。
        """
        if len(messages) <= target_count:
            return list(messages)

        total = len(messages)
        scorer = importance_scorer or self.default_importance_score

        # ---- 第一步:把配对的消息合并为不可拆的选取单元 ----
        units = self._build_selection_units(messages)

        # ---- 第二步:按单元最高分贪心装填,大小计入条数预算 ----
        scored_units = []
        for unit in units:
            scored = [(i, scorer(messages[i], i, total)) for i in unit]
            unit_score = max(s for _, s in scored)
            scored_units.append((unit_score, sorted(unit)))
        scored_units.sort(key=lambda x: x[0], reverse=True)

        selected: set = set()
        budget_left = target_count
        for _, unit in scored_units:
            if len(unit) <= budget_left:
                selected.update(unit)
                budget_left -= len(unit)
            if budget_left == 0:
                break

        # ---- 第三步:按原顺序拼接;配对修复作为最后的安全网 ----
        kept = [messages[i] for i in sorted(selected)]
        return repair_tool_pairs(kept)

    @staticmethod
    def _build_selection_units(messages: List[Message]) -> List[List[int]]:
        """把消息按 tool_use/tool_result 关系分组为选取单元

        通过 tool_use_id 关联:调用所在消息与结果所在消息(可以不相邻)
        归入同一单元;无配对关系的消息各自成组。并查集实现。
        """
        parent = list(range(len(messages)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # 路径压缩
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        tool_use_msg_of = {}  # tool_use_id -> 调用所在消息的下标
        for i, msg in enumerate(messages):
            for tu in msg.get_tool_calls():
                tool_use_msg_of[tu.id] = i

        for i, msg in enumerate(messages):
            for tr in msg.get_tool_results():
                j = tool_use_msg_of.get(tr.tool_use_id)
                if j is not None and j != i:
                    union(i, j)

        groups: dict = {}
        for i in range(len(messages)):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    @staticmethod
    def default_importance_score(msg: Message, index: int, total_count: int) -> float:
        """默认的重要性评分(权重合计约 1.0)

          - 时间新近度(权重 0.5):越靠近当前时刻越重要;
          - 内容权重(最高 0.3):带工具调用的 assistant 消息是
            执行轨迹的关键节点;长文本消息按长度给分(封顶 0.3,
            避免超长消息靠长度霸占预算);
          - 初始上下文(权重 0.2):第一条消息承载任务背景。
        """
        # 最后的消息最重要(权重 0.5)
        recency_score = index / max(total_count - 1, 1) * 0.5

        # 包含工具调用的消息重要(权重 0.3);普通消息按文本长度给分(封顶 0.3)
        if msg.role is MessageRole.ASSISTANT and msg.has_tool_calls():
            content_score = 0.3
        else:
            content_score = min(len(msg.get_text()) / 1000 * 0.15, 0.3)

        # 第一条消息很重要(权重 0.2)
        initial_score = 0.2 if index == 0 else 0.0

        return recency_score + content_score + initial_score
