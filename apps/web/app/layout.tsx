import type { Metadata } from "next";
import "./globals.css";
import UserMenu from "../src/components/UserMenu";

export const metadata: Metadata = {
  title: "Trợ lý nghề nghiệp AI",
  description: "Nền tảng phân tích CV, gợi ý việc làm và cải thiện hồ sơ nghề nghiệp.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body>
        <div className="app-scaffold">
          <UserMenu />
          <main className="container stack" style={{ paddingTop: "120px" }}>
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
