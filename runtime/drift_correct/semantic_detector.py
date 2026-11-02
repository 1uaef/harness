# Author : atg
# Time   :  2026/11/1 11:02

"""漂移检测机制二:语义相似度检测(文档 4.5.2)

关键词检测的盲区:模型在讨论相关话题但用词不同(同义改写),
或者反过来,用词相近但话题已经走偏。语义检测把目标和最近
消息各自编码为向量,用余弦相似度衡量"语义上还在不在做这件事"。

相似度 < 阈值(文档默认 0.6)判定漂移。
"""

import math
import re
from typing import List, Protocol, Tuple

from runtime.message_types import Message


class EmbeddingModel(Protocol):
    """嵌入模型协议:真实实现对接 embedding API(如 OpenAI embeddings)"""

    async def embed(self, text: str) -> List[float]:
        """把文本编码为向量"""
        ...


class HashingEmbedder:
    """词袋哈希嵌入:无外部依赖的测试替身

    把每个词哈希到固定维度的桶里做词袋向量。真实场景请换成
    embedding API——协议兼容,检测器代码不用改。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    async def embed(self, text: str) -> List[float]:
        import hashlib

        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        return vec


class SemanticDriftDetector:
    """使用语义相似度的漂移检测"""

    def __init__(self, embedding_model: EmbeddingModel, similarity_threshold: float = 0.6):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold  # 相似度低于该值判定漂移

    async def detect_semantic_drift(
        self,
        original_goal: str,
        recent_messages: List[Message],
        window: int = 10,
    ) -> Tuple[bool, float]:
        """使用语义相似度检测漂移

        返回: (is_drifted, similarity)
        """
        # 原始目标的嵌入(每次调用都重新编码;生产中目标不变,可缓存)
        goal_embedding = await self.embedding_model.embed(original_goal)

        # 最近消息内容的嵌入
        recent_text = " ".join(msg.get_text() for msg in recent_messages[-window:])
        recent_embedding = await self.embedding_model.embed(recent_text)

        # 余弦相似度
        similarity = self.cosine_similarity(goal_embedding, recent_embedding)

        # 相似度低于阈值表示可能漂移
        is_drifted = similarity < self.similarity_threshold

        return is_drifted, similarity

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度(零向量安全:返回 0)"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a ** 2 for a in vec1))
        norm2 = math.sqrt(sum(a ** 2 for a in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
