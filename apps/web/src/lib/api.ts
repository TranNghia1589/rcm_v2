const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type GuestIntent = "auto" | "score" | "recommend" | "improve_cv" | "general";

export type AccountApiUser = {
  user_id: number;
  email: string;
  full_name: string;
};

export type AuthTokenResponse = {
  access_token: string;
  token_type: string;
  user: AccountApiUser;
};

export type ChatHistoryMessage = {
  role: string;
  content: string;
  created_at: string;
};

export type ChatHistoryLatestResponse = {
  session_id: string | null;
  title: string | null;
  messages: ChatHistoryMessage[];
};

export type GuestAnalyzeResponse = {
  intent: GuestIntent;
  snapshot: {
    target_role: string;
    experience_years: string;
    skills: string[];
    projects_count: number;
  };
  score: {
    overall_score: number;
    grade: string;
    subscores: Record<string, number>;
  };
  recommendations: Array<{
    rank: number;
    job_title: string;
    compatibility_percent: number;
    job_description: string;
    job_url: string;
  }>;
  improve_suggestions: Array<{
    skill: string;
    why: string;
  }>;
};

export type ChatAskResponse = {
  answer: string;
  sources: Array<{
    chunk_id: number;
    document_id: number;
    title: string;
    distance: number;
  }>;
  retrieval_count: number;
  used_fallback: boolean;
  fallback_reason: string;
  fallback_stage: string;
};
async function parseOrThrow(res: Response): Promise<any> {
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  return res.json();
}

export async function registerAccount(params: {
  fullName: string;
  email: string;
  password: string;
}): Promise<AuthTokenResponse> {
  const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      full_name: params.fullName,
      email: params.email,
      password: params.password,
    }),
  });
  return (await parseOrThrow(res)) as AuthTokenResponse;
}

export async function loginAccount(params: {
  email: string;
  password: string;
}): Promise<AuthTokenResponse> {
  const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: params.email, password: params.password }),
  });
  return (await parseOrThrow(res)) as AuthTokenResponse;
}

export async function getMe(token: string): Promise<AccountApiUser> {
  const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });
  return (await parseOrThrow(res)) as AccountApiUser;
}

export async function getLatestChatHistory(token: string): Promise<ChatHistoryLatestResponse> {
  const res = await fetch(`${API_BASE}/api/v1/chat/history/latest`, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });
  return (await parseOrThrow(res)) as ChatHistoryLatestResponse;
}

export async function saveChatTurn(params: {
  token: string;
  sessionId?: string | null;
  title?: string;
  userMessage: string;
  assistantMessage: string;
}): Promise<{ session_id: string; saved: boolean }> {
  const res = await fetch(`${API_BASE}/api/v1/chat/history/save-turn`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${params.token}`,
    },
    body: JSON.stringify({
      session_id: params.sessionId ?? null,
      title: params.title ?? null,
      user_message: params.userMessage,
      assistant_message: params.assistantMessage,
    }),
  });
  return (await parseOrThrow(res)) as { session_id: string; saved: boolean };
}

export async function analyzeGuestCV(params: {
  file: File;
  question: string;
  intent: GuestIntent;
  topK?: number;
  context?: string;
}): Promise<GuestAnalyzeResponse> {
  const form = new FormData();
  form.append("file", params.file);
  let effectiveQuestion = params.question || "";
  
  // Include context in question if provided (for multi-turn conversation awareness)
  if (params.context) {
    effectiveQuestion = `${params.context}\n\nCâu hỏi: ${effectiveQuestion}`;
  }
  
  form.append("question", effectiveQuestion);
  form.append("intent", params.intent);
  form.append("top_k", String(params.topK ?? 5));

  const res = await fetch(`${API_BASE}/api/v1/guest/analyze-cv`, {
    method: "POST",
    body: form,
  });

  return (await parseOrThrow(res)) as GuestAnalyzeResponse;
}

export async function askChat(params: {
  question: string;
  cvId?: number;
  topK?: number;
}): Promise<ChatAskResponse> {
  const res = await fetch(`${API_BASE}/api/v1/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: params.question,
      cv_id: params.cvId ?? null,
      top_k: params.topK ?? 5,
    }),
  });
  return (await parseOrThrow(res)) as ChatAskResponse;
}

