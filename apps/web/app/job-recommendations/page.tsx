"use client";

import Link from "next/link";
import { FormEvent, useMemo, useState } from "react";
import { analyzeGuestCV, GuestAnalyzeResponse } from "../../src/lib/api";

export default function JobRecommendationsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("Gợi ý việc làm phù hợp với CV này");
  const [result, setResult] = useState<GuestAnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => !!file && !loading, [file, loading]);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError("Vui lòng chọn tệp CV");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await analyzeGuestCV({ file, question, intent: "recommend", topK: 8 });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Yêu cầu thất bại");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container stack">
      <section className="panel">
        <h1>Gợi ý việc làm</h1>
        <p className="muted">Tải CV mới và nhập yêu cầu để nhận danh sách việc làm phù hợp.</p>
        <p className="small">
          Mở rộng: <Link href="/chatbot">Trang phân tích CV đầy đủ</Link>
        </p>
      </section>

      <section className="panel">
        <form onSubmit={onSubmit} className="stack">
          <div className="field">
            <label>Tệp CV</label>
            <input
              className="file"
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="field">
            <label>Nội dung yêu cầu</label>
            <textarea
              className="textarea"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ví dụ: Gợi ý việc làm Data Analyst tại Hà Nội"
            />
          </div>

          <div>
            <button className="btn" type="submit" disabled={!canSubmit}>
              {loading ? "Đang gợi ý..." : "Gợi ý việc làm"}
            </button>
          </div>
        </form>
      </section>

      {error ? <div className="error">{error}</div> : null}

      {result ? (
        <>
          <section className="panel">
            <h2>Tóm tắt CV</h2>
            <div className="kv">
              <div className="kv-key">Vai trò mục tiêu</div>
              <div className="kv-value">{result.snapshot.target_role}</div>
            </div>
            <div className="kv">
              <div className="kv-key">Số năm kinh nghiệm</div>
              <div className="kv-value">{result.snapshot.experience_years}</div>
            </div>
            <div className="kv">
              <div className="kv-key">Số dự án</div>
              <div className="kv-value">{result.snapshot.projects_count}</div>
            </div>
          </section>

          <section className="panel">
            <h2>Danh sách việc làm đề xuất</h2>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Vị trí</th>
                    <th>Khoảng cách</th>
                    <th>Lý do</th>
                  </tr>
                </thead>
                <tbody>
                  {result.recommendations.map((r) => (
                    <tr key={`${r.rank}-${r.job_title}`}>
                      <td>{r.rank}</td>
                      <td>{r.job_title}</td>
                      <td>{r.distance}</td>
                      <td>{r.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </main>
  );
}
