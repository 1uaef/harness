# Author : atg
# Time   :  2026/11/1 10:34

"""流式工具执行器(文档 4.3.2,Claude Code 方案)

在响应流中解析出完整工具调用后,由本执行器并发执行:

    - 并发:每个工具调用一个 asyncio.Task,单个慢工具不阻塞
      其他工具的结果收集(对比批处理的"全部完成才能继续");
    - 限流:asyncio.Semaphore 限制最大并发数,防止一次
      发起几十个工具调用打爆下游资源;
    - 错误隔离:单个工具失败不影响其他工具,异常被转换为
      is_error=True 的 ToolResultBlock 回传给模型修正;
    - 进度可观测:每个调用维护 ToolProgress 状态。

注意适配点:文档示例假设 tool.call 返回裸结果;本项目的
Tool 接口(core/tool.py)返回 ToolResult 数据类,带 success
标志和 execution_time,这里按 success 展开成成功/错误两种块。
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from runtime.message_types import AppState, ToolResultBlock, ToolUseBlock
from tools.register import ToolRegistry


class ToolNotFoundError(Exception):
    """注册表中不存在该工具"""


class PermissionDeniedError(Exception):
    """权限检查未通过"""


class ToolStatus(str, Enum):
    """单个工具调用的执行状态"""

    PENDING = "pending"  # 已创建任务,等待信号量
    RUNNING = "running"  # 正在执行
    COMPLETED = "completed"  # 执行成功
    FAILED = "failed"  # 执行失败(含未找到/无权限)


@dataclass
class ToolProgress:
    """工具执行进度"""

    tool_use_id: str
    tool_name: str
    status: ToolStatus = ToolStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def is_done(self) -> bool:
        return self.status in (ToolStatus.COMPLETED, ToolStatus.FAILED)


class StreamingToolExecutor:
    """流式工具执行器:在响应流中处理工具调用"""

    def __init__(self, tool_registry: ToolRegistry, max_concurrent: int = 5):
        self.tool_registry = tool_registry
        self.max_concurrent = max_concurrent  # 并发上限,防止打爆下游
        self._pending_executions: Dict[str, asyncio.Task] = {}  # tool_use_id -> 执行任务
        self._progress: Dict[str, ToolProgress] = {}  # tool_use_id -> 执行进度

    async def execute_tools(
        self,
        tool_uses: List[ToolUseBlock],
        app_state: AppState,
    ) -> Dict[str, ToolResultBlock]:
        """并发执行多个工具调用,返回所有结果

        使用 asyncio.Semaphore 限制并发数;gather(return_exceptions=True)
        等待全部完成——异常不在这里抛出,而是按 tool_use_id 归位成
        错误结果块,保证每个调用都能拿到一个结果与模型配对。
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []

        for tool_use in tool_uses:
            task = asyncio.create_task(self._execute_single_tool(tool_use, app_state, semaphore))
            tasks.append(task)
            self._pending_executions[tool_use.id] = task
            # 记录初始进度
            self._progress[tool_use.id] = ToolProgress(
                tool_use_id=tool_use.id, tool_name=tool_use.name
            )

        # 并发等待所有工具完成(异常也作为结果返回,不中断其他工具)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 组织结果:每个 tool_use_id 必须对应一个 ToolResultBlock
        tool_results: Dict[str, ToolResultBlock] = {}
        for tool_use, result in zip(tool_uses, results):
            if isinstance(result, Exception):
                # 任务级异常(理论上 _execute_single_tool 已兜底,这里是最后防线)
                tool_results[tool_use.id] = ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=f"Tool execution error: {result}",
                    is_error=True,
                    error_type=type(result).__name__,
                )
            else:
                tool_results[tool_use.id] = result

        # 任务已全部完成,清理 pending 表;进度记录保留供查询
        for tool_use in tool_uses:
            self._pending_executions.pop(tool_use.id, None)

        return tool_results

    async def _execute_single_tool(
        self,
        tool_use: ToolUseBlock,
        app_state: AppState,
        semaphore: asyncio.Semaphore,
    ) -> ToolResultBlock:
        """执行单个工具:查表 -> 权限检查 -> 执行 -> 组织结果"""
        progress = self._progress[tool_use.id]
        async with semaphore:  # 超过 max_concurrent 的任务在这里排队
            try:
                # 1. 查找工具
                tool = self.tool_registry.get(tool_use.name)
                if not tool:
                    raise ToolNotFoundError(f"Tool '{tool_use.name}' not found")

                # 2. 权限检查(上下文传入应用状态,工具可按会话做细粒度控制)
                if not tool.check_permissions(app_state):
                    raise PermissionDeniedError(f"Permission denied for tool '{tool_use.name}'")

                # 3. 执行工具
                progress.status = ToolStatus.RUNNING
                progress.started_at = datetime.now(timezone.utc)
                result = await tool.call(tool_use.input)

                # 4. 组织结果(按项目 Tool 接口的 success 标志分流)
                progress.status = ToolStatus.COMPLETED if result.success else ToolStatus.FAILED
                progress.finished_at = datetime.now(timezone.utc)
                return ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=not result.success,
                    error_type=result.error_type,
                )

            except Exception as e:
                # 单个工具的一切失败都降级为错误结果块,不向上传播:
                # 其他并发中的工具不受影响,模型也能拿到错误描述自行修正
                progress.status = ToolStatus.FAILED
                progress.finished_at = datetime.now(timezone.utc)
                return ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=str(e),
                    is_error=True,
                    error_type=type(e).__name__,
                )

    def get_tool_progress(self, tool_use_id: str) -> Optional[ToolProgress]:
        """获取某个工具的执行进度(不存在时返回 None)"""
        return self._progress.get(tool_use_id)

    @property
    def pending_count(self) -> int:
        """仍在执行中的工具调用数量"""
        return len(self._pending_executions)
