"use client";

import Link from "next/link";
import { FormEvent, useMemo, useState } from "react";
import { analyzeGuestCV, GuestAnalyzeResponse, GuestIntent } from "../../src/lib/api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  ts: string;
  data?: GuestAnalyzeResponse;
};

const QUICK_PROMPTS = [
  "Chấm điểm CV của tôi",
  "Gợi ý việc làm phù hợp với CV này",
  "Tôi cần cải thiện kỹ năng gì để ứng tuyển tốt hơn?",
];

function gradeClass(grade: string): string {
  const g = (grade || "").toUpperCase();
  if (g === "A") return "badge badge-a";
  if (g === "B") return "badge badge-b";
  if (g === "C") return "badge badge-c";
  return "badge badge-d";
}

function intentToLabel(intent: GuestIntent): string {
  if (intent === "score") return "Chấm điểm CV";
  if (intent === "recommend") return "Gợi ý việc làm";
  if (intent === "improve_cv") return "Cải thiện CV";
  return "Tự động";
}

function buildAssistantText(result: GuestAnalyzeResponse): string {
  const parts: string[] = [];
  parts.push(`Đã xử lý yêu cầu: ${intentToLabel(result.intent)}.`);
  parts.push(`Điểm CV hiện tại: ${result.score.overall_score} (mức ${result.score.grade}).`);

  if (result.recommendations.length > 0) {
    const top = result.recommendations
      .slice(0, 3)
      .map((x) => x.job_title)
      .join(", ");
    parts.push(`Top vị trí phù hợp: ${top}.`);
  }

  if (result.improve_suggestions.length > 0) {
    const top = result.improve_suggestions
      .slice(0, 3)
      .map((x) => x.skill)
      .join(", ");
    parts.push(`Nên ưu tiên cải thiện: ${top}.`);
  }

  parts.push("Bạn có thể tiếp tục đặt câu hỏi để phân tích sâu hơn.");
  return parts.join(" ");
}

export default function ChatbotPage() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("Tôi muốn gợi ý công việc phù hợp với CV này");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [latestResult, setLatestResult] = useState<GuestAnalyzeResponse | null>(null);

  const canSubmit = useMemo(() => !!file && !loading && !!question.trim(), [file, loading, question]);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError("Vui lòng chọn tệp CV (.pdf/.docx/.txt)");
      return;
    }

    const userText = question.trim();
    if (!userText) {
      setError("Vui lòng nhập nội dung yêu cầu");
      return;
    }

    setLoading(true);
    setError("");

    const userMsg: ChatMessage = {
      id: `${Date.now()}-u`,
      role: "user",
      text: userText,
      ts: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setQuestion("");

    try {
      // Always auto-detect from user question to avoid conflict with manual intent selection.
      const data = await analyzeGuestCV({ file, question: userText, intent: "auto", topK: 5 });
      setLatestResult(data);
      const botMsg: ChatMessage = {
        id: `${Date.now()}-a`,
        role: "assistant",
        text: buildAssistantText(data),
        ts: new Date().toLocaleTimeString(),
        data,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Yêu cầu thất bại");
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setLatestResult(null);
    setError("");
  };

  return (
    <main className="container stack">
      <section className="panel">
        <h1>Chat tư vấn CV 1-1</h1>
        <p className="muted">Lịch sử chat chỉ lưu trong phiên hiện tại. Tải lại trang sẽ tự xóa.</p>
        <p className="small">
          Truy cập nhanh: <Link href="/job-recommendations">Trang gợi ý việc làm</Link>
        </p>
      </section>

      <section className="panel stack">
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
          <label>Gợi ý nhanh</label>
          <div className="quick-actions">
            {QUICK_PROMPTS.map((p) => (
              <button key={p} type="button" className="quick-btn" onClick={() => setQuestion(p)}>
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="panel" style={{ background: "#f8fbff" }}>
          <div className="section-title">
            <h2>Lịch sử hội thoại</h2>
            <button className="btn" type="button" onClick={clearChat} style={{ padding: "6px 10px" }}>
              Xóa lịch sử
            </button>
          </div>
          <div style={{ display: "grid", gap: 10 }}>
            {messages.length === 0 ? <p className="muted">Chưa có hội thoại. Hãy nhập câu hỏi đầu tiên.</p> : null}
            {messages.map((m) => (
              <div
                key={m.id}
                style={{
                  background: m.role === "user" ? "#e8f1ff" : "#ffffff",
                  border: "1px solid #d8e3f6",
                  borderRadius: 10,
                  padding: 10,
                }}
              >
                <div style={{ fontWeight: 700, marginBottom: 6 }}>
                  {m.role === "user" ? "Bạn" : "Trợ lý"} <span className="muted">· {m.ts}</span>
                </div>
                <div>{m.text}</div>
              </div>
            ))}
          </div>
        </div>

        <form onSubmit={onSubmit} className="stack">
          <div className="field">
            <label>Nội dung yêu cầu</label>
            <textarea
              className="textarea"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ví dụ: So sánh CV của tôi với vị trí Data Analyst và gợi ý cải thiện"
            />
          </div>

          <div>
            <button className="btn" type="submit" disabled={!canSubmit}>
              {loading ? "Đang phân tích..." : "Gửi câu hỏi"}
            </button>
          </div>
        </form>
      </section>

      {error ? <div className="error">{error}</div> : null}

      {latestResult ? (
        <>
          <section className="panel">
            <div className="section-title">
              <h2>Tóm tắt CV</h2>
              <span className="badge badge-intent">Loại yêu cầu: {intentToLabel(latestResult.intent)}</span>
            </div>
            <div className="kv">
              <div className="kv-key">Vai trò mục tiêu</div>
              <div className="kv-value">{latestResult.snapshot.target_role}</div>
            </div>
            <div className="kv">
              <div className="kv-key">Số năm kinh nghiệm</div>
              <div className="kv-value">{latestResult.snapshot.experience_years}</div>
            </div>
            <div className="kv">
              <div className="kv-key">Số dự án</div>
              <div className="kv-value">{latestResult.snapshot.projects_count}</div>
            </div>
            <div style={{ marginTop: 10 }}>
              <div className="kv-key" style={{ marginBottom: 8 }}>
                Kỹ năng trích xuất
              </div>
              <div className="chips">
                {latestResult.snapshot.skills.length > 0 ? (
                  latestResult.snapshot.skills.map((s) => (
                    <span className="chip" key={s}>
                      {s}
                    </span>
                  ))
                ) : (
                  <span className="muted">Chưa trích xuất được kỹ năng</span>
                )}
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="section-title">
              <h2>Điểm CV</h2>
              <span className={gradeClass(latestResult.score.grade)}>Mức {latestResult.score.grade}</span>
            </div>
            <div className="kv">
              <div className="kv-key">Điểm tổng</div>
              <div className="kv-value">{latestResult.score.overall_score}</div>
            </div>
            <div className="table-wrap" style={{ marginTop: 10 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Tiêu chí</th>
                    <th>Điểm</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(latestResult.score.subscores).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {latestResult.recommendations.length > 0 ? (
            <section className="panel">
              <h2>Danh sách việc làm gợi ý</h2>
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
                    {latestResult.recommendations.map((r) => (
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
          ) : null}

          {latestResult.improve_suggestions.length > 0 ? (
            <section className="panel">
              <h2>Gợi ý cải thiện CV</h2>
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Kỹ năng</th>
                      <th>Vì sao cần bổ sung</th>
                    </tr>
                  </thead>
                  <tbody>
                    {latestResult.improve_suggestions.map((it) => (
                      <tr key={it.skill}>
                        <td>{it.skill}</td>
                        <td>{it.why}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </>
      ) : null}
    </main>
  );
}
