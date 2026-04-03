"use client";

import { FormEvent, useState } from "react";
import { HybridRecommendResponse, recommendHybrid } from "@/lib/api";

export default function JobRecommendationsPage() {
  const [cvId, setCvId] = useState(1);
  const [question, setQuestion] = useState("Goi y viec lam phu hop");
  const [result, setResult] = useState<HybridRecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const data = await recommendHybrid(question, cvId);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ padding: 24 }}>
      <h1>Hybrid Recommendation</h1>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12, maxWidth: 700 }}>
        <label>
          CV ID
          <input
            type="number"
            value={cvId}
            onChange={(e) => setCvId(Number(e.target.value))}
            style={{ width: "100%", padding: 8 }}
          />
        </label>

        <label>
          Question
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            style={{ width: "100%", padding: 8 }}
          />
        </label>

        <button type="submit" disabled={loading} style={{ width: 180, padding: 10 }}>
          {loading ? "Loading..." : "Recommend"}
        </button>
      </form>

      {error ? <p style={{ color: "crimson" }}>{error}</p> : null}
      {result ? <pre style={{ marginTop: 16, whiteSpace: "pre-wrap" }}>{JSON.stringify(result, null, 2)}</pre> : null}
    </main>
  );
}
