import os
import getpass
from typing import Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from rag import load_retriever
from settings import graphconfig

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("enter your key: ")

# load vector stores
vec_retr, bm_retr = load_retriever()

# load reranker
reranker = CrossEncoder(graphconfig.rerank.model)

def rerank_docs(
    docs, 
    query: str,
    topk: int = 5
):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    # rerank
    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )
    return [doc for _, doc in ranked[:topk]]

def remov_dup(
    doc1: list,
    doc2: list,
    method: str = "overlap",
    threshold: float = 0.8
):
    removed_li = []
    if method == "overlap":
        removed_set = []
        for i in doc1 + doc2:
            if len(removed_li) == 0:
                removed_li.append(i)
                removed_set.append(set(i.page_content.split()))
            else:
                doc_set = set(i.page_content.split())
                trigger = False
                for kept in removed_set:
                    score = len(doc_set & kept) / min(len(doc_set), len(kept))
                    if score >= threshold:
                        trigger = True
                        break
                if not trigger:
                    removed_li.append(i)
                    removed_set.append(doc_set)
    return removed_li

class Overall_State(TypedDict):
    input: str
    translated: str
    retrieved_msg: str
    final_res: str

# translate input into English
def rewrite_node(
    state: Overall_State
) -> Overall_State:
    llm = ChatOpenAI(
        model = graphconfig.llm.rewrite_node
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                你是一个用于英文文档检索的查询重写助手。
                用户会用中文提问，但知识库文档是英文（爬行动物/爬宠相关）。

                你的任务：
                - 把用户问题改写成最适合英文检索的短查询
                - 保留核心生物学 / 饲养 / 医疗术语
                - 优先使用 reptile care / veterinary / husbandry 领域常见英文表达
                - 如果涉及具体物种（如蜥蜴、蛇等），尽量用英文名称
                - 不要解释，不要回答，只输出一个英文检索查询

                要求：
                - 输出尽量简洁
                - 只输出字符串
                """
            ),
            (
                "human",
                "input: {input}"
            )
        ]
    )
    parser = StrOutputParser()
    lc = prompt | llm | parser
    state["translated"] = lc.invoke(
        {"input": state["input"]}
    )
    return state

# load vector stores and pick relative content
def retr_node(
    state: Overall_State
) -> Overall_State:
    vec_res = vec_retr.invoke(state["translated"])
    bm_res = bm_retr.invoke(state["translated"])

    # remove overlap
    removed_res = remov_dup(
        doc1 = vec_res,
        doc2 = bm_res,
        method = graphconfig.removedup.method,
        threshold = graphconfig.removedup.threshold
    )

    ranked_res = rerank_docs(
        docs = removed_res,
        query = state["translated"],
        topk = graphconfig.rerank.topk
    )
    state["retrieved_msg"] = f"\n\n".join(
        f"source = {doc.metadata.get('source', '')}\n{doc.page_content}"
        for doc in ranked_res
    )
    return state

# answer users' question based on retrieved content
def ans_node(
    state: Overall_State
) -> Overall_State:
    llm = ChatOpenAI(
        model = graphconfig.llm.ans_node
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                你是一位专业的爬行动物（reptile）饲养与兽医知识顾问。

                请严格根据给定的英文资料回答用户的中文问题。

                要求：
                - 用中文回答
                - 只根据提供的资料作答，不要补充资料中没有的内容
                - 如果资料不足以回答，就明确说“根据现有资料，我无法确定”
                - 回答应专业、准确，符合 reptile husbandry / veterinary 的表达习惯
                - 如果涉及：
                    * 饲养（温度、湿度、光照等）→ 给出明确参数或范围（若资料中有）
                    * 疾病/症状 → 清晰说明可能原因，但避免超出资料推断
                    * 行为 → 结合生态或习性解释
                - 回答结构尽量清晰（可以分点），但不要啰嗦

                指示：
                - 只返回字符串形式
                """
            ),
            (
                "human",
                """
                questions:
                {input}
                 
                context:
                {context}
                """
            )
        ]
    )
    parser = StrOutputParser()
    lc = prompt | llm | parser
    state["final_res"] = lc.invoke(
        {
            "input": state["input"],
            "context": state["retrieved_msg"]
        }
    )
    return state

# build graph
builder = StateGraph(
    Overall_State,
)
builder.add_node("translate", rewrite_node)
builder.add_node("retriever", retr_node)
builder.add_node("answer", ans_node)

builder.add_edge(START, "translate")
builder.add_edge("translate", "retriever")
builder.add_edge("retriever", "answer")
builder.add_edge("answer", END)

if __name__ == "__main__":

    INPUT = "Ackie Monitor的温度湿度要求是怎么样的"

    graph = builder.compile()

    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    
    init_state = Overall_State(
        input = INPUT
    )
    res = graph.invoke(
        init_state
    )
    print(res["final_res"])
