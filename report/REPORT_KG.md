# Báo cáo Day 19 — Flat RAG vs GraphRAG

**Họ tên:** Vũ Văn Hà  **MSSV:** 2A202602589  **Ngày:** 5/10/2026

> Kỳ vọng và thang điểm: `SUBMISSION.md`. Mọi số liệu phải khớp với `ket_qua_benchmark_kg.txt`. Bản thiết kế ontology nộp riêng ở `report/ONTOLOGY.md`.

**Cấu hình lần chạy:** chat `gemini-3.5-flash-lite`, embedding `gemini-embedding-2`, `top_k=3`, `chunk_size=800`, 176 chunk, KG 201 node / 378 cạnh.

## 1. Chi phí (10 điểm)

```
== Indexing (one-off)
pipeline  calls    in_tok  out_tok       USD  seconds
flat          4 ~   45870        0   0.00000     75.4
graph        24 ~   80489     5536   0.00568    135.0

== Querying (mean per question)
pipeline  recall  judge   in_tok  out_tok       USD  seconds
flat        0.22   0.50      649       22   0.00007     6.92
graph       0.93   1.83     4208      184   0.00049     2.23
```

Dấu `~`: `in_tok` của indexing là **ước lượng** (~2,5 ký tự/token) vì endpoint OpenAI-compatible của Gemini trả `usage = null` cho `/embeddings`. Phần chat có `usage` thật nên không đánh dấu.

| Chỉ số | Flat | Graph | Graph / Flat |
| --- | --- | --- | --- |
| Indexing USD | $0.00000 | $0.00568 | **×∞** (Flat = 0) |
| Indexing giây | 75,4 | 135,0 | ×1,79 |
| Indexing in_tok | ~45.870 | ~80.489 | ×1,75 |
| Indexing calls | 4 | 24 | ×6,0 |
| Mỗi câu: USD | $0.00007 | $0.00049 | ×7,0 |
| Mỗi câu: in_tok | 649 | 4.208 | **×6,5** |
| Mỗi câu: giây | 6,92 | 2,23 | **×0,32** |

**Chi phí tăng thêm đến từ đâu?**

1. **Dựng graph, không phải embedding.** `graph_index = flat_index + kg_build`: Flat chỉ embed 176 chunk (4 lần gọi, free tier nên $0), còn Graph cộng thêm 20 lần gọi LLM trong `build_graph` để trích xuất `Case`/`Person`/`Crime`/`Substance` từ 20 bài báo → **$0.00568, chi phí duy nhất là chat**. Bỏ điều kiện này thì GraphRAG "rẻ ngang" Flat RAG.
2. **Prompt dài hơn 6,5 lần khi trả lời.** `GraphRAGAgent.answer` ghép thêm dữ kiện `context()` vào prompt (Điều luật, khoản, khung phạt). Đây là cái giá phải trả để đổi recall 0,22 → 0,93.
3. **Độ trễ lại GIẢM 3,1 lần** (6,92s → 2,23s). Nguyên nhân: con số 35,16s của Q5 flat là do một lần gọi bị 429 phải chờ retry, không phải chi phí thuần của Flat RAG. Loại nhiễu này thì hai bên gần như ngang độ trễ.

**Điểm hòa vốn:** mỗi câu Graph tốn thêm ~$0.00042 và ~3.559 token. Với giá chat hiện tại, điểm hòa vốn về **chi phí** phải trên ~1.200 câu hỏi — về mặt tiền thì GraphRAG không hòa vốn ở quy mô lab. Điểm hòa vốn thực sự nằm ở **chất lượng**: một câu sai mất ~15 phút tra cứu thủ công thì chỉ ~30 câu đã bù đủ.

## 2. Từng câu hỏi (10 điểm)

| Câu | Loại | Flat recall / judge | Graph recall / judge | Thắng | Vì sao |
| --- | --- | --- | --- | --- | --- |
| Q1 | single-hop-law | 1.00 / 2 | 1.00 / 2 | Hòa | Đáp án nằm ngay trong một chunk luật; vector search đủ, không cần graph |
| Q2 | single-hop-news | 0.00 / 0 | 1.00 / 2 | **Graph** | Flat trả lời "Không đủ thông tin" vì top-3 chunk không chứa đủ tên; graph đi thẳng `Person -[:INVOLVED_IN]-> Case` |
| Q3 | cross-kb | 0.33 / 1 | 1.00 / 2 | **Graph** | Flat có "36 tháng" nhưng không nối được sang Điều 251; graph đi qua `Crime` cầu nối |
| Q4 | cross-kb | 0.00 / 0 | 1.00 / 2 | **Graph** | Cần ghép "bị bắt về hành vi gì" + "khung phạt tối đa"; graph lấy đủ khoản 4 Điều 255 |
| Q5 | cross-kb-multi-hop | 0.00 / 0 | 0.60 / 1 | **Graph** (nhưng sai) | Graph sai Điều 250 → Điều 248; xem lỗi E2 ở mục 3 |
| Q6 | aggregation | 0.00 / 0 | 1.00 / 2 | **Graph** | Cần gom nhiều vụ; graph quét `Case -[:INVOLVES]-> Substance` |

**Quy luật:** Flat RAG chỉ khá ở Q1 — câu hỏi *single-hop-law* mà đáp án nằm sẵn trong một chunk. Trên mọi câu cần **điều kiện** (`cross-kb`) hoặc **gộp nhiều nguồn** (`aggregation`), Flat RAG rơi về 0.00 và báo "Không đủ thông tin". Càng nhiều bước nhảy giữa 2 KB, lợi thế của GraphRAG càng rõ.

## 3. Phân tích lỗi (20 điểm)

### Lỗi E2: `context()` chọn sai Điều luật vì không lọc theo tội danh của vụ án

- **Hiện tượng:** Q5 hỏi về vụ vận chuyển MDMA của Cái Quang Huy. Đáp án chuẩn là **Điều 250 khoản 4**, nhưng GraphRAG trả lời **Điều 248 khoản 4**. `recall = 0.60`, `judge = 1`.

- **Bằng chứng:** câu trả lời trong `ket_qua_benchmark_kg.txt`, dòng `--- Q5 [cross-kb-multi-hop] graph`:

  > *"Khoản và Điều luật: Khoản 4 **Điều 248** Bộ luật Hình sự (quy dịnh về tội **sản xuất** trái phép chất ma túy, hoặc áp dụng theo khung tương ứng của tội vận chuyển ma túy có khối lượng lớn đối với MDMA từ 100 gam trở lên)."*

  So sánh `data/benchmark_kg.json` Q5: đáp án chuẩn là *"tội vận chuyển trái phép chất ma túy (**Điều 250 BLHS**) ... thuộc khoản 4 Điều 250"*. Model tự nhận ra mâu thuẫn và phải viết *"hoặc áp dụng theo khung tương ứng"* — dấu hiệu nó **đoán** vì graph không cho dữ kiện chắc chắn.

  Truy vấn chứng minh tội danh đúng là `vận chuyển`, không phải `sản xuất`:

  ```cypher
  MATCH (k:Case)-[:CHARGED_WITH]->(c:Crime)
  WHERE k.name CONTAINS 'Nội Bài' RETURN k.name AS vu, c.name AS toi_danh;
  ```

  ```
  vu: Vụ vận chuyển ma túy qua sân bay Nội Bài liên quan đến Cái Quang Huy
  toi_danh: vận chuyển trái phép chất ma túy      ← Điều 250
  ```

  Và số khoản luật `MENTIONS` MDMA — cho thấy "MDMA từ 100 gam trở lên" xuất hiện ở **18 khoản** thuộc nhiều Điều khác nhau, nên lọc theo chất là chưa đủ:

  ```cypher
  MATCH (c:Clause)-[:MENTIONS]->(s:Substance {name:'MDMA'})
  RETURN c.id AS khoan ORDER BY khoan;
  ```

  → 18 khoản rải rác trên nhiều Điều (Điều 248, 249, 250, 251, 252…). `context()` đưa cả 18 vào prompt nên LLM không biết chọn cái nào.

- **Nguyên nhân:** nằm ở **thiết kế ontology + Cypher ở KG-3**, không phải ở prompt hay LLM. Quan hệ `(:Crime)<-[:DEFINES]-(:Article)` **đã có đủ** thông tin để lọc chính xác: tội `vận chuyển trái phép chất ma túy` chỉ được `Article: 'Điều 250 BLHS'` định nghĩa. Nhưng `context()` khi lọc khoản chỉ dùng điều kiện *"chất có trong câu hỏi"* (`MENTIONS Substance`), **không** dùng điều kiện *"Điều luật do chính tội danh của vụ định nghĩa"*. Bỏ lỡ một đường đi đã có sẵn trong graph → LLM phải tự chọn trong 18 khoản và chọn sai.

- **Đề xuất sửa:** thêm một nhánh `UNION` vào Cypher của `context()`, ưu tiên khoản thuộc Điều luật do tội danh của vụ định nghĩa:

  ```cypher
  MATCH (k:Case)-[:CHARGED_WITH]->(:Crime)<-[:DEFINES]-(:Article)-[:HAS_CLAUSE]->(cl:Clause)
  WHERE elementId(k) IN $ids
  RETURN cl.id, cl.number, cl.penalty, cl.text
  UNION
  // chỉ khi không có vụ nào khớp tội danh mới rơi xuống lọc theo chất
  ...
  ```

  **Đánh đổi:** thêm một vòng truy vấn nên chậm hơn một chút; nhưng prompt **ngắn lại** vì bỏ được phần lớn khoản nhiễu. Đúng với điểm "Đánh đổi cần nghĩ" ở Bước 5: lấy ít khoản nhưng chính xác hơn.

### Lỗi E1: Cầu nối `Crime` gãy khi LLM không trả tội danh về tên chuẩn

- **Hiện tượng:** một số `Case` không có cạnh `CHARGED_WITH`, nên **mất hoàn toàn** đường đi sang KB luật. Câu hỏi chạm tới các vụ đó chỉ còn vector search — tức mất lợi thế của chính GraphRAG.

- **Bằng chứng:** sau `python bench_kg.py --judge`, truy vấn:

  ```cypher
  MATCH (k:Case) WHERE NOT (k)-[:CHARGED_WITH]->() RETURN k.name, k.doc_id;
  ```

  ```
  Vụ vận chuyển ma túy qua sân bay Nội Bài của Cái Quang Huy        | news-100260918080821054
  Vụ vận chuyển hơn 800kg chất nghi là ma túy và vũ khí tại Preah Sihanouk | news-100260924145818945
  Triệt phá chuyên án A3-626P                                      | news-100261002184934505
  ```

  Cả 3 **lẽ ra phải nối được**: bài báo có nêu rõ hành vi vận chuyển ma túy. Bài đầu thậm chí là nguồn của Q5/Q6, và cùng một người (`Cái Quang Huy`) lại xuất hiện ở một vụ khác **có** cạnh `CHARGED_WITH` → chứng minh đây là lỗi trích xuất ngẫu nhiên của LLM theo lần gọi, không phải bản chất bài báo.

- **Nguyên nhân:** ở **KG-2 `build_graph`**. Prompt trích xuất đã đưa danh sách tội danh chuẩn, nhưng LLM vẫn tự đặt tên không khớp (ví dụ `"vận chuyển ma túy trái phép"`, `"triệt phá đường dây"`), rồi `link_entity` với `cutoff=0.8` không đủ gần nên trả `None` → không tạo cạnh. Đúng là điểm yếu đã ghi trong bản thiết kế: *"`Crime` khóa theo tên do LLM tự đặt"*. Vì node `Crime` `MERGE` theo tên chuẩn nên một tên lệch sẽ tạo node mới thay vì tái dùng node cũ.

- **Đề xuất sửa:** không nới `cutoff` (dễ nối sai, mà nối sai còn tệ hơn không nối), mà ghi fallback: khi `link_entity` trả `None` cho tội danh, **tạo `Crime` node mới với `name` nguyên văn LLM trả về** và gắn cờ `unlinked = true`. Câu hỏi về vụ đó vẫn đi được sang phía "sự kiện", và có thể đo chất lượng trích xuất offline bằng `MATCH (c:Crime {unlinked:true}) RETURN c.name`. **Đánh đổi:** thêm nhiễu cho Cypher, phải lọc lại khi truy vấn.

### Lỗi E4: `recall` đo bằng từ khóa nên mâu thuẫn với `judge`

- **Hiện tượng:** Q5 graph đạt `judge = 1` nhưng `recall = 0.60`; câu trả lời chứa từ đúng nhưng đặt sai ngữ cảnh — nhắc "Điều 248" (sai) nhưng cũng nhắc "vận chuyển", "MDMA" (đúng).

- **Bằng chứng:** `must_include` của Q5 gồm `["vận chuyển", "MDMA", "Điều 250", "khoản 4", "tử hình"]`. Câu trả lời chứa 3/5 từ nên `recall = 0.60`, nhưng thiếu `Điều 250` và `tử hình` — tức bỏ sót đúng phần cốt lõi. Ngược lại câu trả lời chứa **"Điều 248"** — không hề có trong `must_include`, nên `recall` **không hề phạt** cho câu sai đó.

- **Nguyên nhân:** nằm ở **phép đo** (`keyword_recall` trong `bench_kg.py:46`), không phải pipeline. Cơ chế này chỉ kiểm "từ có xuất hiện hay không", không kiểm từ đó **đúng chỗ nào**. Câu trả lời toàn sai mà vô tình chứa đủ từ khóa vẫn đạt điểm; câu đúng ý nhưng diễn đạt khác vẫn bị trừ.

- **Đề xuất sửa:** giữ `recall` làm chỉ báo **rẻ và khách quan** nhưng **không kết luận về độ đúng chỉ từ nó** — phải đọc cặp `recall` + `judge` như ở Q5. Nếu muốn chặt hơn: thêm nhóm từ khóa **phủ định** (ví dụ `["không phải Điều 248"]`) hoặc chấm bằng embedding similarity với `gold`. **Đánh đổi:** phức tạp hơn, và `judge` (LLM) lại tốn token — vốn đã là chi phí ưu tiên lớn ở mục 1.

## 4. Kết luận (5 điểm)

**Khi nào nên dùng KG:** khi câu hỏi cần **đi qua ranh giới giữa 2 KB** hoặc **gộp nhiều tài liệu**. Bằng chứng từ lần chạy này: GraphRAG đạt recall **1.00** trên cả Q2, Q3, Q4, Q6 trong khi Flat RAG đạt **0.00** trên những câu đó. Nguyên nhân đúng như thiết kế: `Crime` là node cầu nối, chỉ khi đi qua nó mới nối được người trong tin với Điều luật.

**Khi nào Flat RAG là đủ:** khi câu hỏi **single-hop** và đáp án nằm sẵn trong một chunk. Q1 là bằng chứng: Flat đạt recall 1.00 / judge 2, **bằng** Graph, dùng 649 token so với 4.208 (tiết kiệm 6,5 lần). Thêm nữa, nếu dữ liệu không có quan hệ rõ ràng (chỉ là tập văn bản độc lập) thì KG không tạo thêm giá trị gì.

**Điều kiện cụ thể để quyết định:**
- **< ~30 câu/ngày, đa số single-hop** → Flat RAG. Chi phí và độ trễ thấp hơn rõ rệt.
- **Số câu lớn, có câu cần đối chiếu luật ↔ vụ việc** → GraphRAG. Vấn đề duy nhất là chất lượng truy hồi, **không phải** tiền: mỗi câu chỉ $0.00049.
- **Chi phí dựng ban đầu $0.00568** trả một lần cho toàn bộ vòng đời; dùng lại index nhiều lần thì hòa vốn rất nhanh.

**Điều kiện tiên quyết:** GraphRAG chỉ thắng khi **KG đúng**. Lỗi E2 cho thấy một điều kiện Cypher thiếu đã làm Q5 giảm từ recall 1.00 xuống 0.60. Cải thiện `context()` có giá trị cao hơn nhiều so với đổi sang model đắt hơn — và gần như **miễn phí**.

## 5. Tự kiểm (5 điểm)

```
$ pytest tests/test_base.py -q
41 passed in 0.27s

$ pytest tests/test_graph.py -q
7 passed in 0.10s
```

`tests/test_llm_batch.py` (16 test em thêm cho batch embedding + retry 429) tách riêng để `test_base.py` giữ nguyên 41 test gốc của đề.

```
$ python bench_kg.py --check
[OK] Dữ liệu: 18 điều luật, 20 bài báo
[OK] KG-1 link_entity
[provider] chat = gemini:gemini-3.5-flash-lite | embedding = gemini:gemini-embedding-2
[OK] KG-2 build_graph: 148 node / 293 cạnh, đường xuyên 2 KB dài 2 cạnh
[OK] KG-3 context: 18 dữ kiện, có Điều 251
[OK] KG-4 GraphRAGAgent.answer
[OK] Chi phí check: 1 lần gọi LLM, $0.00000.
```

Ảnh Neo4j: `report/img/kg_count.png`, `report/img/kg_cross_kb.png`, `report/img/kg_my_case.png`.
Người đã chọn cho `kg_my_case.png`: **Cái Quang Huy**.

## Vấn đề gặp phải (không tính điểm)

**1. Lỗi 429 quota Gemini khi chạy benchmark.**

Lệnh: `python bench_kg.py --judge`

```
openai.RateLimitError: Error code: 429 - Quota exceeded for metric:
generativelanguage.googleapis.com/embed_content_free_tier_requests, limit: 100, model: gemini-embedding-1.0
Please retry in 38.9s.
```

Nguyên nhân: `EmbeddingStore` gọi embedding **mỗi chunk một lần**. Corpus 38 tài liệu → **176 chunk** → 176 lần gọi, vượt hạn mức 100 lần/phút của free tier.

Cách đã sửa trong `src/llm.py`:
- Thêm `MeteredLLM.embed_batch()` gom `EMBED_BATCH_SIZE` (mặc định 50) chunk vào **một** request. Indexing 176 chunk tốn **4 call** thay vì 176.
- Thêm `_with_retry()` phân biệt 2 loại quota dùng chung mã 429: quota **theo phút** (chờ rồi thử lại) và quota **theo ngày** (fail-fast, vì provider yêu cầu chờ ~18h thì retry vô nghĩa).

**2. `gemini-2.5-flash-lite` không dùng được với key mới.**

```
openai.NotFoundError: Error code: 404 - This model models/gemini-2.5-flash-lite
is no longer available to new users.
```

Đã đổi sang `gemini-3.5-flash-lite` (giá $0.10/$0.40 mỗi 1M token, có sẵn trong `src/llm.py`). Với tài khoản mới, các model còn dùng được: `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`.

**3. `.env` bị biến môi trường PowerShell ghi đè.**

Đã sửa `.env` sang `gemini-3.5-flash-lite` nhưng `bench_kg.py` vẫn in `gemini-2.5-flash-lite`. Nguyên nhân: session PowerShell đã có sẵn `$env:GEMINI_CHAT_MODEL`, mà `bench_kg.py` dùng `load_dotenv(override=False)` nên **biến môi trường thắng `.env`**. Xử lý: `$env:GEMINI_CHAT_MODEL=$null` trước khi chạy.

**4. `gemini-embedding-2` trả `usage = null`, không đo được token.**

Nếu để 0 thì dòng `Indexing` của Flat ghi `in_tok = 0`, khiến chi phí indexing của Flat trông rẻ hơn thật và **làm lệch so sánh với GraphRAG**. Đã thêm `estimate_tokens()` (ước lượng ~2,5 ký tự/token vì tiếng Việt nhiều âm tiết hơn tiếng Anh) và đánh dấu bằng `~` trong file kết quả để không thể nhầm là số đo thật.

**5. Lỗi khác gặp khi chạy thật:** Gemini trả `data[index] = null` (OpenAI trả số thứ tự) → đã xử lý bằng fallback về thứ tự mảng; log tiếng Việt trong handler retry gây `UnicodeEncodeError` trên console cp1252 → đã đổi sang ASCII để không che mất lỗi quota thật.
