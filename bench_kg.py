"""Flat RAG vs GraphRAG (Neo4j) on the two drug knowledge bases: accuracy, latency, tokens, USD.

    docker run -d --name neo4j-drug-kg -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5
    python bench_kg.py --build --limit 2   # load law + 2 news articles into Neo4j with YOUR build_graph (KG-2)
    python bench_kg.py --build             # load both full KBs (~20 LLM calls, ~0.01 USD)
    python bench_kg.py --check    # self-check KG-1..KG-4 on 1 news article (~1 LLM call, < 0.001 USD)
    python bench_kg.py            # needs an API key in .env (OpenAI, OpenRouter, Gemini or Anthropic — see src/llm.py)
    python bench_kg.py --judge    # + LLM-as-judge score (metered separately, not counted in pipeline cost)

Writes ket_qua_benchmark_kg.txt (summary table + every answer).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv

PACKAGE = os.getenv("LAB_SOLUTION_PACKAGE", "src")
_m = import_module(PACKAGE)
graph_mod = import_module(f"{PACKAGE}.graph")
llm_mod = import_module(f"{PACKAGE}.llm")
Document, EmbeddingStore, KnowledgeBaseAgent, RecursiveChunker = (
    _m.Document, _m.EmbeddingStore, _m.KnowledgeBaseAgent, _m.RecursiveChunker)
Usage = llm_mod.Usage

JUDGE_PROMPT = """Chấm câu trả lời so với đáp án chuẩn. Trả về JSON {{"score": 0|1|2, "reason": "..."}}:
2 = đúng và đủ các ý chính, 1 = đúng một phần, 0 = sai hoặc không trả lời được.
Câu hỏi: {question}
Đáp án chuẩn: {gold}
Câu trả lời: {answer}"""

def chunk_docs(docs: list, chunk_size: int) -> list:
    chunker = RecursiveChunker(chunk_size=chunk_size)
    return [
        Document(id=f"{doc.id}#{i}", content=piece, metadata={**doc.metadata, "doc_id": doc.id})
        for doc in docs
        for i, piece in enumerate(chunker.chunk(doc.content))
    ]

def keyword_recall(answer: str, keywords: list[str]) -> float:
    return sum(k.lower() in answer.lower() for k in keywords) / len(keywords)

def metered(llm, fn):
    """Run fn(), return (result, Usage delta incl. wall-clock seconds)."""
    before, start = llm.usage, time.perf_counter()
    result = fn()
    delta = llm.usage - before
    delta.seconds = time.perf_counter() - start
    return result, delta

def fail(code: str, problem: str, fix: str) -> None:
    print(f"\n[LỖI {code}] {problem}\n  Cách sửa: {fix}\n  Tra bảng lỗi: LAB_GUIDE.md mục 'Xử lý lỗi'")
    raise SystemExit(1)

def ok(message: str) -> None:
    print(f"[OK] {message}")

def make_llm():
    try:
        llm = llm_mod.MeteredLLM()
    except (RuntimeError, ImportError) as error:
        fail("SETUP-1", f"Chưa dùng được provider LLM: {error}",
             "copy .env.example thành .env, điền ít nhất một key: OPENAI_API_KEY, OPENROUTER_API_KEY, "
             "GEMINI_API_KEY hoặc ANTHROPIC_API_KEY (Anthropic chỉ dùng cho chat; embedding cần một trong 3 key đầu).")
    print(f"[provider] chat = {llm.chat_model} | embedding = {llm.embedding_model}")
    return llm

def connect_graph():
    from neo4j.exceptions import AuthError, ServiceUnavailable

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    try:
        return graph_mod.Neo4jGraph(uri, os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
    except ServiceUnavailable:
        fail("SETUP-2", f"Không kết nối được Neo4j tại {uri}.",
             "bật Docker Desktop, chạy `docker start neo4j-drug-kg` (lần đầu: lệnh docker run ở LAB_GUIDE.md Bước 0), "
             "đợi ~20 giây rồi chạy lại.")
    except AuthError:
        fail("SETUP-3", "Neo4j từ chối đăng nhập.",
             "NEO4J_USER/NEO4J_PASSWORD trong .env phải khớp NEO4J_AUTH lúc docker run (mặc định neo4j/password123).")

def load_corpus():
    law_docs = graph_mod.load_markdown_docs("data/drug_law")
    news_docs = graph_mod.load_markdown_docs("data/drug_news")
    if not law_docs or not news_docs:
        fail("DATA-1", "Thiếu dữ liệu trong data/drug_law hoặc data/drug_news.",
             "python scripts/crawl_drug_corpus.py --news-limit 20")
    return law_docs, news_docs

CHECK_NEWS = "news-100260918080821054"   # Lê Minh Thành — mua bán trái phép chất ma túy (Điều 251)
CHECK_QUESTION = "Lê Minh Thành bị tuyên bao nhiêu tháng tù, về tội gì, theo Điều nào của Bộ luật Hình sự?"

def check() -> int:
    """Ontology-independent self-check: only the contract in src/graph.py is tested, not your labels."""
    law_docs, news_docs = load_corpus()
    ok(f"Dữ liệu: {len(law_docs)} điều luật, {len(news_docs)} bài báo")
    crimes = ["mua bán trái phép chất ma túy", "vận chuyển trái phép chất ma túy"]
    if graph_mod.link_entity("Tội Mua bán trái phép chất ma tuý", crimes) != crimes[0]             or graph_mod.link_entity("lừa đảo chiếm đoạt tài sản", crimes) is not None:
        fail("KG-1", "link_entity chưa map đúng biến thể chính tả, hoặc nối bừa tên không liên quan.",
             "pytest tests/test_graph.py -k LinkEntity -v")
    ok("KG-1 link_entity")

    graph = connect_graph()
    ok("Neo4j kết nối được")
    llm = make_llm()
    news = [d for d in news_docs if d.id == CHECK_NEWS]
    graph.reset()
    graph_mod.build_graph(graph, law_docs, news, llm.chat)
    by_doc = {r["doc_id"]: r["n"] for r in graph.run(
        "MATCH (n) WHERE n.doc_id IS NOT NULL RETURN n.doc_id AS doc_id, count(n) AS n")}
    law_ids = [d.id for d in law_docs]
    if not any(i in by_doc for i in law_ids) or CHECK_NEWS not in by_doc:
        fail("KG-2", "build_graph chưa tạo node mang property doc_id cho cả 2 KB.",
             "mỗi node sinh ra từ 1 tài liệu phải có doc_id = Document.id (hợp đồng đầu file src/graph.py).")
    path = graph.run(
        "MATCH (a), (b) WHERE a.doc_id IN $law AND b.doc_id = $news "
        "MATCH p = shortestPath((a)-[*..4]-(b)) RETURN length(p) AS hops ORDER BY hops LIMIT 1",
        law=law_ids, news=CHECK_NEWS)
    if not path:
        fail("KG-2", "Không có đường đi (<= 4 cạnh) nối node của KB luật với node của bài báo: cầu nối 2 KB bị gãy.",
             "xem node cầu nối trong ontology của bạn; thử trong Neo4j Browser: "
             f"MATCH (b {{doc_id:'{CHECK_NEWS}'}})-[*..2]-(x) RETURN b, x")
    stats = graph.stats()
    ok(f"KG-2 build_graph: {stats['nodes']} node / {stats['relationships']} cạnh, "
       f"đường xuyên 2 KB dài {path[0]['hops']} cạnh")

    facts = graph.context(CHECK_QUESTION, [CHECK_NEWS])
    if not any("251" in f for f in facts):
        fail("KG-3", "Neo4jGraph.context không đưa được Điều 251 BLHS vào dữ kiện cho câu hỏi về Lê Minh Thành.",
             "in thử graph.context(...) và viết lại Cypher multi-hop trong Neo4j Browser (LAB_GUIDE.md Bước 5).")
    ok(f"KG-3 context: {len(facts)} dữ kiện, có Điều 251")

    store = EmbeddingStore(collection_name="check", embedding_fn=_m._mock_embed)
    store.add_documents([Document("check", "Lê Minh Thành bị tuyên 36 tháng tù.", {"doc_id": CHECK_NEWS})])
    prompt = graph_mod.GraphRAGAgent(store=store, graph=graph, llm_fn=lambda p: p).answer(CHECK_QUESTION, top_k=1)
    if "251" not in prompt or "36 tháng" not in prompt:
        fail("KG-4", "GraphRAGAgent.answer chưa đưa cả dữ kiện graph và đoạn văn bản vào prompt.",
             "pytest tests/test_graph.py -k GraphRAGAgent -v")
    ok("KG-4 GraphRAGAgent.answer")
    graph.close()
    ok(f"Chi phí check: {llm.usage.calls} lần gọi LLM, ${llm.usage.usd:.5f}. "
       "Graph nhỏ (luật + 1 bài) vẫn còn trong Neo4j để bạn xem; chạy --judge để dựng graph đầy đủ.")
    return 0

def build(limit: int | None) -> int:
    """Only load the KG (KG-2) so you can inspect it in Neo4j Browser — no embeddings, no questions."""
    law_docs, news_docs = load_corpus()
    if limit:  # always include the article the guide's example queries use (Lê Minh Thành)
        news_docs = sorted(news_docs, key=lambda d: d.id != CHECK_NEWS)[:limit]
    graph = connect_graph()
    llm = make_llm()
    graph.reset()
    _, usage = metered(llm, lambda: graph_mod.build_graph(graph, law_docs, news_docs, llm.chat))
    stats = graph.stats()
    print(f"Đã nạp {len(law_docs)} điều luật + {len(news_docs)} bài báo: "
          f"{stats['nodes']} node / {stats['relationships']} cạnh "
          f"({usage.calls} lần gọi LLM, ${usage.usd:.5f}, {usage.seconds:.1f}s)")
    for row in graph.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
        print(f"  node  {row['label']:<20} {row['n']}")
    for row in graph.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC"):
        print(f"  cạnh  {row['rel']:<20} {row['n']}")
    shared = [r["label"] for r in graph.run(
        "MATCH (n) WHERE n.doc_id IS NULL RETURN DISTINCT labels(n)[0] AS label")]
    if shared:
        print(f"  Label không có doc_id: {', '.join(shared)} "
              "(chỉ hợp lệ nếu là node dùng chung giữa nhiều tài liệu, ví dụ tội danh, chất)")
    graph.close()
    print("Mở http://localhost:7474 để xem graph (LAB_GUIDE.md Bước 8.1).")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--out", default="ket_qua_benchmark_kg.txt")
    parser.add_argument("--check", action="store_true", help="self-check KG-1..KG-4 (~1 LLM call)")
    parser.add_argument("--build", action="store_true", help="only load the KG into Neo4j (KG-2)")
    parser.add_argument("--limit", type=int, help="with --build: number of news articles to load")
    args = parser.parse_args()
    load_dotenv(override=False)
    if args.check:
        return check()
    if args.build:
        return build(args.limit)

    law_docs, news_docs = load_corpus()
    connect_graph().close()  # fail fast before paying for embeddings
    llm = make_llm()
    chunks = chunk_docs(law_docs + news_docs, args.chunk_size)
    questions = json.loads(Path("data/benchmark_kg.json").read_text(encoding="utf-8"))

    # --- Indexing. Flat RAG = embed chunks. GraphRAG = the same vector index + KG build.
    # batch_embedding_fn: ~ceil(len(chunks)/EMBED_BATCH_SIZE) API calls instead of one per chunk,
    # which keeps a few hundred chunks inside the provider's per-minute embedding quota.
    print(f"[index] embedding {len(chunks)} chunk qua {llm.embedding_model} "
          f"(lô {llm_mod.EMBED_BATCH_SIZE}/request)...", flush=True)
    store = EmbeddingStore(collection_name="drug_kb", embedding_fn=llm.embed,
                           batch_embedding_fn=llm.embed_batch)
    _, flat_index = metered(llm, lambda: store.add_documents(chunks))

    graph = connect_graph()
    graph.reset()
    _, kg_build = metered(llm, lambda: graph_mod.build_graph(graph, law_docs, news_docs, llm.chat))
    graph_index = flat_index + kg_build

    # --- Querying.
    flat_agent = KnowledgeBaseAgent(store=store, llm_fn=llm.chat)
    graph_agent = graph_mod.GraphRAGAgent(store=store, graph=graph, llm_fn=llm.chat)
    rows = []
    for q in questions:
        for name, agent in (("flat", flat_agent), ("graph", graph_agent)):
            answer, usage = metered(llm, lambda: agent.answer(q["question"], top_k=args.top_k))
            row = {"id": q["id"], "type": q["type"], "pipeline": name, "answer": answer, "usage": usage,
                   "recall": keyword_recall(answer, q["must_include"])}
            if args.judge:
                verdict = llm.chat(JUDGE_PROMPT.format(question=q["question"], gold=q["gold"], answer=answer), json_mode=True)
                row["judge"] = json.loads(verdict).get("score", 0)
            rows.append(row)
            print(f"{q['id']} {name:5} recall={row['recall']:.2f} {usage.seconds:.2f}s ${usage.usd:.5f}")
    stats = graph.stats()
    graph.close()

    # --- Report.
    lines = [f"Chat model: {llm.chat_model} | Embedding: {llm.embedding_model} | top_k={args.top_k} "
             f"| chunk_size={args.chunk_size} | chunks={len(chunks)} | KG: {stats['nodes']} nodes / {stats['relationships']} rels", ""]
    lines.append("== Indexing (one-off)")
    lines.append(f"{'pipeline':8} {'calls':>6} {'in_tok':>9} {'out_tok':>8} {'USD':>9} {'seconds':>8}")
    for name, u in (("flat", flat_index), ("graph", graph_index)):
        mark = "~" if u.est_input_tokens else " "
        lines.append(f"{name:8} {u.calls:>6} {mark}{u.input_tokens:>8} {u.output_tokens:>8} {u.usd:>9.5f} {u.seconds:>8.1f}")
    if flat_index.est_input_tokens:
        lines.append(f"~ = in_tok uoc luong (~{llm_mod.CHARS_PER_TOKEN} ky tu/token) vi provider khong tra usage cho "
                     f"{llm.embedding_model}; chi co dung o dong nay, khong phai so do thuc.")
    lines += ["", "== Querying (mean per question)"]
    lines.append(f"{'pipeline':8} {'recall':>7} {'judge':>6} {'in_tok':>8} {'out_tok':>8} {'USD':>9} {'seconds':>8}")
    for name in ("flat", "graph"):
        mine = [r for r in rows if r["pipeline"] == name]
        n = len(mine)
        total = sum((r["usage"] for r in mine), Usage())
        judge = f"{sum(r.get('judge', 0) for r in mine) / n:.2f}" if args.judge else "-"
        lines.append(f"{name:8} {sum(r['recall'] for r in mine) / n:>7.2f} {judge:>6} {total.input_tokens / n:>8.0f} "
                     f"{total.output_tokens / n:>8.0f} {total.usd / n:>9.5f} {total.seconds / n:>8.2f}")
    lines += ["", "== Per question"]
    for r in rows:
        lines.append(f"--- {r['id']} [{r['type']}] {r['pipeline']} recall={r['recall']:.2f}"
                     f"{' judge=' + str(r['judge']) if 'judge' in r else ''} {r['usage'].seconds:.2f}s")
        lines.append(r["answer"].strip())
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: lines.index("== Per question")]))
    print(f"Saved {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
