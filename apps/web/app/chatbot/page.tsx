"use client";

import Link from "next/link";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { getAccount } from "../../src/lib/account";
import {
  analyzeGuestCV,
  askChat,
  getLatestChatHistory,
  GuestAnalyzeResponse,
  GuestIntent,
  saveChatTurn,
} from "../../src/lib/api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  ts: string;
};

const QUICK_PROMPTS = [
  "Hãy chấm điểm CV của tôi",
  "CV này cần cải thiện điểm nào trước?",
  "Gợi ý các kỹ năng tôi cần bổ sung để cải thiện CV",
];

function intentToLabel(intent: GuestIntent | "general"): string {
  if (intent === "score") return "chấm điểm CV";
  if (intent === "recommend") return "gợi ý việc làm";
  if (intent === "improve_cv") return "cải thiện CV";
  return "tư vấn tổng quát";
}

const SUBSCORE_LABELS: Record<string, string> = {
  skills: "Kỹ năng",
  experience: "Kinh nghiệm",
  projects: "Dự án",
  certifications: "Chứng chỉ",
  education: "Học vấn",
};

function summarizeStrengthWeakness(subscores: Record<string, number>): {
  strengths: string[];
  weaknesses: string[];
} {
  const entries = Object.entries(subscores || {});
  if (entries.length === 0) return { strengths: [], weaknesses: [] };

  const sortedDesc = [...entries].sort((a, b) => b[1] - a[1]);
  const sortedAsc = [...entries].sort((a, b) => a[1] - b[1]);

  const strengths = sortedDesc.slice(0, 2).map(([k, v]) => `${SUBSCORE_LABELS[k] ?? k}: ${v}/100`);
  const weaknesses = sortedAsc.slice(0, 2).map(([k, v]) => `${SUBSCORE_LABELS[k] ?? k}: ${v}/100`);
  return { strengths, weaknesses };
}

function buildScoreReply(result: GuestAnalyzeResponse): string {
  const lines: string[] = [];
  lines.push(`Mình đã ${intentToLabel("score")} cho CV của bạn. Điểm hiện tại là ${result.score.overall_score}/100 (${result.score.grade}).`);

  if (result.snapshot.target_role && result.snapshot.target_role !== "Unknown") {
    lines.push(`Đánh giá theo vai trò mục tiêu: ${result.snapshot.target_role}.`);
  }

  const { strengths, weaknesses } = summarizeStrengthWeakness(result.score.subscores);
  if (strengths.length > 0) {
    lines.push(`Điểm mạnh: ${strengths.join("; ")}.`);
  }
  if (weaknesses.length > 0) {
    lines.push(`Điểm cần cải thiện: ${weaknesses.join("; ")}.`);
  }

  lines.push("Nếu muốn, bạn có thể hỏi thêm để mình phân tích sâu cách cải thiện theo role bạn nhắm tới.");
  return lines.join("\n\n");
}

function buildQuestionForChat(result: GuestAnalyzeResponse, userText: string): string {
  const targetRole = result.snapshot.target_role || "Unknown";
  const exp = result.snapshot.experience_years || "Unknown";
  const skills = (result.snapshot.skills || []).slice(0, 15).join(", ") || "Unknown";
  const hintIntent = intentToLabel((result.intent as GuestIntent | "general") || "general");

  return [
    `Intent phat hien: ${hintIntent}.`,
    `CV snapshot: target_role=${targetRole}; experience_years=${exp}; skills=${skills}.`,
    `Cau hoi nguoi dung: ${userText}`,
  ].join("\n");
}


function compactForDisplay(text: string): string {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/\n\s*\n/g, "\n");
}
function formatTime(input: string): string {
  const d = new Date(input);
  if (Number.isNaN(d.getTime())) return new Date().toLocaleTimeString();
  return d.toLocaleTimeString();
}

export default function ChatbotPage() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [showNewMessageHint, setShowNewMessageHint] = useState(false);

  const formRef = useRef<HTMLFormElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const prevMessageCountRef = useRef(0);

  const canSubmit = useMemo(() => !!file && !loading && !!question.trim(), [file, loading, question]);

  const isNearBottom = () => {
    const el = chatHistoryRef.current;
    if (!el) return true;
    const threshold = 28;
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
  };

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    const el = chatHistoryRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
  };

  useEffect(() => {
    const syncAccount = async () => {
      const acc = getAccount();
      const nextToken = acc?.token ?? null;
      setToken(nextToken);
      setSessionId(null);
      setMessages([]);
      setShowNewMessageHint(false);
      stickToBottomRef.current = true;

      if (!nextToken) return;
      try {
        const hist = await getLatestChatHistory(nextToken);
        setSessionId(hist.session_id ?? null);
        const mapped: ChatMessage[] = hist.messages
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m, idx) => ({
            id: `hist-${idx}-${m.created_at}`,
            role: m.role as "user" | "assistant",
            text: m.role === "assistant" ? compactForDisplay(m.content) : m.content,
            ts: formatTime(m.created_at),
          }));
        setMessages(mapped);
      } catch {
        // ignore
      }
    };

    syncAccount();
    window.addEventListener("rcm-account-changed", syncAccount);
    return () => window.removeEventListener("rcm-account-changed", syncAccount);
  }, []);

  useEffect(() => {
    const currentCount = messages.length;
    const prevCount = prevMessageCountRef.current;

    if (currentCount > prevCount) {
      if (stickToBottomRef.current) {
        requestAnimationFrame(() => scrollToBottom("smooth"));
      } else {
        setShowNewMessageHint(true);
      }
    }

    prevMessageCountRef.current = currentCount;
  }, [messages]);

  const onHistoryScroll = () => {
    const nearBottom = isNearBottom();
    stickToBottomRef.current = nearBottom;
    if (nearBottom) setShowNewMessageHint(false);
  };

  const onJumpToLatest = () => {
    stickToBottomRef.current = true;
    setShowNewMessageHint(false);
    scrollToBottom("smooth");
  };

  const onComposerKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSubmit) formRef.current?.requestSubmit();
    }
  };

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError("Vui lòng tải CV trước khi chat.");
      return;
    }

    const userText = question.trim();
    if (!userText) {
      setError("Vui lòng nhập câu hỏi.");
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
      const data = await analyzeGuestCV({ file, question: userText, intent: "auto", topK: 5 });
      let answerText = "";

      if (data.intent === "score") {
        answerText = buildScoreReply(data);
      } else {
        const chatQuestion = buildQuestionForChat(data, userText);
        const chat = await askChat({ question: chatQuestion, topK: 6 });
        answerText = chat.answer || "Mình chưa có đủ dữ liệu để trả lời chính xác. Bạn có thể mô tả chi tiết hơn mục tiêu công việc và kỹ năng hiện có nhé.";

        if (chat.used_fallback) {
          answerText += "\n\nMình cần thêm thông tin để trả lời sát hơn: role mục tiêu, số năm kinh nghiệm, kỹ năng chính và mức lương mong muốn.";
        }
      }

      answerText = compactForDisplay(answerText);

      const botMsg: ChatMessage = {
        id: `${Date.now()}-a`,
        role: "assistant",
        text: answerText,
        ts: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, botMsg]);

      if (token) {
        const saved = await saveChatTurn({
          token,
          sessionId,
          title: "Chat CV 1:1",
          userMessage: userText,
          assistantMessage: answerText,
        });
        setSessionId(saved.session_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Yêu cầu thất bại");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container stack">
      <section className="panel">
        <h1>Chatbot tư vấn CV 1:1</h1>
        <p className="muted">Chat theo dạng tin nhắn. Khi đăng nhập, lịch sử hội thoại sẽ lưu trong hệ thống theo tài khoản của bạn.</p>
        <p className="small">
          Cần danh sách job chi tiết? <Link href="/job-recommendations">Mở tab Gợi ý việc làm</Link>
        </p>
      </section>

      <section className="panel stack">
        <div className="field">
          <label>Tệp CV</label>
          <input className="file" type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
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

        <div className="chat-shell">
          <div ref={chatHistoryRef} className="chat-history" onScroll={onHistoryScroll}>
            {messages.length === 0 ? (
              <div className="chat-empty">Chưa có hội thoại. Hãy gửi tin nhắn đầu tiên.</div>
            ) : (
              messages.map((m) => (
                <div key={m.id} className={`chat-bubble ${m.role === "user" ? "chat-user" : "chat-assistant"}`}>
                  <div className="chat-meta">
                    <span>{m.role === "user" ? "Bạn" : "Chatbot"}</span>
                    <span>{m.ts}</span>
                  </div>
                  <div className="chat-text">{m.text}</div>
                </div>
              ))
            )}
          </div>

          {showNewMessageHint ? (
            <div className="new-message-bar">
              <button type="button" className="new-message-btn" onClick={onJumpToLatest}>
                Tin nhắn mới ↓
              </button>
            </div>
          ) : null}

          <form ref={formRef} className="chat-composer" onSubmit={onSubmit}>
            <textarea
              className="chat-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onComposerKeyDown}
              rows={1}
              placeholder="Nhập câu hỏi của bạn..."
            />
            <button className="btn" type="submit" disabled={!canSubmit}>
              {loading ? "Đang gửi..." : "Gửi"}
            </button>
          </form>
        </div>
      </section>

      {error ? <div className="error">{error}</div> : null}
    </main>
  );
}



