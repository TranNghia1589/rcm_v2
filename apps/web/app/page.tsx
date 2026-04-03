import Link from "next/link";

const featureCards = [
  {
    title: "Phân tích CV",
    desc: "Tải CV mới, chọn yêu cầu và nhận ngay kết quả chấm điểm, gợi ý việc làm hoặc cải thiện CV.",
    href: "/chatbot",
    cta: "Mở trang phân tích",
  },
  {
    title: "Gợi ý việc làm phù hợp",
    desc: "Tập trung vào gợi ý công việc phù hợp dựa trên CV và yêu cầu thực tế của người dùng.",
    href: "/job-recommendations",
    cta: "Mở trang gợi ý việc làm",
  },
];

export default function HomePage() {
  return (
    <main className="container stack">
      <section className="hero panel">
        <div className="hero-chip">Trợ lý nghề nghiệp AI</div>
        <h1 className="hero-title">Xây dựng lộ trình nghề nghiệp chỉ với một lần tải CV</h1>
        <p className="hero-subtitle">
          Tải CV mới và nhập yêu cầu. Hệ thống tự phân tích và trả về điểm CV, gợi ý việc làm
          hoặc đề xuất cải thiện kỹ năng.
        </p>
        <div className="cta-row">
          <Link className="cta-link primary" href="/chatbot">
            Bắt đầu phân tích đầy đủ
          </Link>
          <Link className="cta-link" href="/job-recommendations">
            Đi đến gợi ý việc làm
          </Link>
        </div>
      </section>

      <section className="feature-grid">
        {featureCards.map((card) => (
          <article key={card.title} className="panel feature-card">
            <h2>{card.title}</h2>
            <p className="muted">{card.desc}</p>
            <Link className="feature-link" href={card.href}>
              {card.cta}
            </Link>
          </article>
        ))}
      </section>

      <section className="panel">
        <div className="section-title">
          <h2>Quy trình sử dụng</h2>
          <span className="badge badge-intent">Luồng phân tích CV</span>
        </div>
        <div className="steps-grid">
          <div className="step-item">
            <div className="step-no">1</div>
            <div>
              <h3>Tải CV mới</h3>
              <p className="muted">Hỗ trợ định dạng .pdf, .docx, .txt.</p>
            </div>
          </div>
          <div className="step-item">
            <div className="step-no">2</div>
            <div>
              <h3>Nhập yêu cầu</h3>
              <p className="muted">Ví dụ: gợi ý việc làm, chấm điểm CV, đề xuất kỹ năng cần cải thiện.</p>
            </div>
          </div>
          <div className="step-item">
            <div className="step-no">3</div>
            <div>
              <h3>Nhận kết quả có cấu trúc</h3>
              <p className="muted">Hiển thị theo thẻ/bảng rõ ràng cho điểm CV, gợi ý việc làm và khoảng thiếu kỹ năng.</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
