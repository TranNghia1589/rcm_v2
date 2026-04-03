# AI Agent Test Basis (Research-Informed)

Bo test co ban trong `tests/` duoc thiet ke theo cac nguyen tac danh gia agent/RAG da duoc cong bo:

## 1) Groundedness va retrieval quality (RAG)
- Nguon: RAG (Lewis et al., 2020), RAGAs (Es et al., 2023/2025 update).
- Ap dung:
  - Kiem tra API chatbot tra ve `sources` va metadata fallback.
  - Kiem tra retrieval layer xu ly `top_k` va query embedding on dinh.

## 2) Reliability khi tool/module loi
- Nguon: ReAct (Yao et al., 2022), Reflexion (Shinn et al., 2023).
- Ap dung:
  - Kiem tra orchestrator van tra duoc ket qua graph khi vector candidate bi loi.
  - Kiem tra warning duoc phat ra de observability.

## 3) Reproducible benchmark-style evaluation
- Nguon: SWE-bench (Jimenez et al., 2024).
- Ap dung:
  - Test contract theo input/output schema ro rang.
  - Fixture deterministic, khong phu thuoc mang/DB that trong test co ban.

## Files da bo sung test
- `tests/integration/test_chatbot_api.py`
- `tests/integration/test_recommendation_api.py`
- `tests/rag/test_retrieve.py`
- `tests/recommendation/test_hybrid_recommender.py`
- `tests/graph/test_cypher_queries.py`
