const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type GuestIntent = "auto" | "score" | "recommend" | "improve_cv";

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
    distance: number;
    reason: string;
  }>;
  improve_suggestions: Array<{
    skill: string;
    why: string;
  }>;
};

export async function analyzeGuestCV(params: {
  file: File;
  question: string;
  intent: GuestIntent;
  topK?: number;
}): Promise<GuestAnalyzeResponse> {
  const form = new FormData();
  form.append("file", params.file);
  form.append("question", params.question || "");
  form.append("intent", params.intent);
  form.append("top_k", String(params.topK ?? 5));

  const res = await fetch(`${API_BASE}/api/v1/guest/analyze-cv`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }

  return (await res.json()) as GuestAnalyzeResponse;
}
