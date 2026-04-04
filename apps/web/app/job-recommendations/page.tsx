"use client";

import { FormEvent, useMemo, useState } from "react";
import { analyzeGuestCV, GuestAnalyzeResponse } from "../../src/lib/api";

const FIXED_QUESTION = "Tìm việc làm phù hợp với CV này";

export default function JobRecommendationsPage() {
  const [file, setFile] = useState<File | null>(null);
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
      const data = await analyzeGuestCV({ file, question: FIXED_QUESTION, intent: "recommend", topK: 10 });
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Yêu cầu thất bại");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container stack">
      <section className="panel">
        <h1>Gợi ý việc làm theo CV</h1>
        <p className="muted">Tải CV lên và bấm nút để nhận danh sách 10 việc làm phù hợp nhất.</p>
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

          <div>
            <button className="btn" type="submit" disabled={!canSubmit}>
              {loading ? "Đang tìm việc làm..." : "Tìm việc làm"}
            </button>
          </div>
        </form>
      </section>

      {error ? <div className="error">{error}</div> : null}

      {result ? (
        <section className="panel">
          <h2>Top 10 việc làm phù hợp</h2>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Vị trí</th>
                  <th>Điểm tương thích</th>
                  <th>Mô tả job</th>
                  <th>Link</th>
                </tr>
              </thead>
              <tbody>
                {result.recommendations.length === 0 ? (
                  <tr>
                    <td colSpan={4}>Chưa tìm thấy kết quả phù hợp.</td>
                  </tr>
                ) : (
                  result.recommendations.map((r) => (
                    <tr key={`${r.rank}-${r.job_title}`}>
                      <td>{r.job_title}</td>
                      <td>{r.compatibility_percent}%</td>
                      <td>{r.job_description}</td>
                      <td>
                        {r.job_url ? (
                          <a href={r.job_url} target="_blank" rel="noreferrer" aria-label={`Mở tin tuyển dụng ${r.job_title}`}>
                            ↗
                          </a>
                        ) : (
                          "-"
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </main>
  );
}
