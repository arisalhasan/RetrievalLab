# Milestone 6 — A reusable search platform

The same Python search engines now serve two interfaces: Streamlit for a person and FastAPI for another program. Neither interface calculates benchmark scores during a search. Experiments remain separate, so UI rendering and HTTP overhead do not distort the recorded search timings.

## Run locally

From the project root, using the existing Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dense,platform]"
.\.venv\Scripts\python.exe -m uvicorn retrieval_lab.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/docs> for the interactive API contract. In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$request = @{query='cell division and mitosis'; system='bm25'; top_k=10} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType application/json -Body $request
```

For macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Only the documented Windows reference environment has been executed here. The final full dependency snapshot is `requirements-release.lock.txt`; use that for exact reference versions rather than resolving extras again.

## Contract and implementation

`GET /health` checks the service without loading models; it lists initialized engines. `GET /systems` lists supported method names. `POST /search` takes a nonblank query of at most 2,000 characters, a system (`bm25`, `dense`, `hybrid`, `rerank`, or `trained`), and an integer `top_k` from 1 to 20. The response contains document IDs, titles, abstracts, scores, configuration, dataset hashes, query time, and available ranking traces.

`api.py` validates the request, while `SearchService` holds lazily initialized core engines. A lock permits one CPU search at a time; another concurrent request receives 429 with `Retry-After: 1`. Invalid input receives 422. Missing local data, embeddings, or weights receives 503 with recovery details in the server log. There is no silent fallback and no model download or corpus encoding inside a request. A first request can initialize an already-prepared index; its initialization time is excluded from `query_time_ms`, but a caller's HTTP time includes it.

Prepare neural search first using the existing `model-download`, `build-dense`, and `reranker-download` commands. The `trained` method additionally needs the selected trained configuration and its checkpoint. Use one worker: more workers duplicate model memory. The API binds to loopback by default. Public hosting, authentication, internet exposure, load balancing, and multiuser service guarantees are outside this local project release.

`exports.py` converts a verified paired comparison into a pandas CSV and matplotlib PNG/SVG. The dashboard offers JSON, CSV, and chart downloads. Quality, time, and memory use separate axes and units; raw similarity/logit scores are never compared across model families.

## Verify this milestone

Start the API and run the request above. Check that the first hit matches a BM25 search in Streamlit. Submit a blank query and verify 422. In **Compare systems**, compare the matching M5 hybrid and reranker runs and download the CSV and figure. Check the values against the JSON. Automated tests exercise real BM25 through HTTP, request validation, busy/missing-asset behavior, the OpenAPI schema, and export values.

Interview questions:

1. Why should a dashboard request and a benchmark experiment use the same retrieval implementation?
2. What is the difference between query inference time, index preparation time, and HTTP response time?
3. Why does this CPU API serialize inference and return 429 rather than start unlimited searches?
