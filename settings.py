from dataclasses import dataclass, field

@dataclass
class SplitterConfig:
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])
    chunk_size: int = 1200
    overlap: int = 100

@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"

@dataclass
class RetrieverConfig:
    search_type: str = "mmr"
    search_kwargs: dict = field(
        default_factory = lambda: {
                "k": 20,
                "lambda_mult": 0.5
            }
    )
    

@dataclass
class BM25Config:
    k: int = 10

@dataclass
class RAGConfig:
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    bm25: BM25Config = field(default_factory=BM25Config)

@dataclass
class RerankConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    topk: int = 5

@dataclass
class RemoveDupConfig:
    method: str = "overlap"
    threshold: float = 0.8

@dataclass
class LLMConfig:
    rewrite_node: str = "gpt-4o"
    ans_node: str = "gpt-5"

@dataclass
class GraphConfig:
    rerank: RerankConfig = field(default_factory=RerankConfig)
    removedup: RemoveDupConfig = field(default_factory=RemoveDupConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


ragconfig = RAGConfig()
graphconfig = GraphConfig()