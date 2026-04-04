"use client";

import { useEffect, useRef, useState } from "react";
import { clearAccount, getAccount, setAccount } from "../lib/account";
import { loginAccount, registerAccount } from "../lib/api";

type MenuItem = {
  label: string;
  icon: string;
  onClick: () => void;
};

type AuthMode = "login" | "register";

export default function UserMenu() {
  const [open, setOpen] = useState(false);
  const [account, setAccountState] = useState<ReturnType<typeof getAccount> | null>(null);
  const [mounted, setMounted] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");

  useEffect(() => {
    // Initialize account state from localStorage after hydration
    setAccountState(getAccount());
    setMounted(true);

    const onClickOutside = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        setAuthOpen(false);
      }
    };

    const onStorage = () => setAccountState(getAccount());
    const onChanged = () => setAccountState(getAccount());

    document.addEventListener("mousedown", onClickOutside);
    document.addEventListener("keydown", onKeydown);
    window.addEventListener("storage", onStorage);
    window.addEventListener("rcm-account-changed", onChanged);

    return () => {
      document.removeEventListener("mousedown", onClickOutside);
      document.removeEventListener("keydown", onKeydown);
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("rcm-account-changed", onChanged);
    };
  }, []);

  const userName = mounted ? (account?.fullName || "Khách") : "Khách";
  const userSubtitle = mounted ? (account?.email || "Chưa đăng nhập") : "Chưa đăng nhập";
  const userStatus = mounted ? (account ? "Đã đăng nhập" : "Khách") : "Khách";

  const openAuth = (mode: AuthMode) => {
    setAuthMode(mode);
    setAuthError("");
    if (mode === "register") {
      setFullName(account?.fullName || "");
    }
    setEmail(account?.email || "");
    setPassword("");
    setAuthOpen(true);
    setOpen(false);
  };

  const onLogout = () => {
    clearAccount();
    setOpen(false);
    setAuthOpen(false);
  };

  const onSubmitAuth = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setAuthError("");

    const emailVal = email.trim();
    const passwordVal = password.trim();
    const fullNameVal = fullName.trim();

    if (!emailVal || !passwordVal) {
      setAuthError("Vui lòng nhập đầy đủ email và mật khẩu.");
      return;
    }
    if (passwordVal.length < 6) {
      setAuthError("Mật khẩu phải có ít nhất 6 ký tự.");
      return;
    }
    if (authMode === "register" && !fullNameVal) {
      setAuthError("Vui lòng nhập họ tên.");
      return;
    }

    setAuthLoading(true);
    try {
      const res =
        authMode === "register"
          ? await registerAccount({ fullName: fullNameVal, email: emailVal, password: passwordVal })
          : await loginAccount({ email: emailVal, password: passwordVal });

      setAccount({
        userId: res.user.user_id,
        fullName: res.user.full_name,
        email: res.user.email,
        token: res.access_token,
      });

      setAuthOpen(false);
      setPassword("");
      setAuthError("");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Xác thực thất bại");
    } finally {
      setAuthLoading(false);
    }
  };

  const menuItems: MenuItem[] = [
    { label: "Trang chủ", icon: "🏠", onClick: () => (window.location.href = "/") },
    { label: "Chatbot", icon: "💬", onClick: () => (window.location.href = "/chatbot") },
    { label: "Gợi ý việc làm", icon: "📌", onClick: () => (window.location.href = "/job-recommendations") },
    ...(account
      ? [{ label: "Đăng xuất", icon: "⏏️", onClick: onLogout }]
      : [
          { label: "Đăng nhập", icon: "🔓", onClick: () => openAuth("login") },
          { label: "Đăng ký", icon: "🆕", onClick: () => openAuth("register") },
        ]),
  ];

  const onToggle = () => setOpen((s) => !s);

  return (
    <>
      <header className="app-header" aria-label="Thanh điều hướng người dùng">
        <div className="header-left">
          <div className="brand">
            <span className="brand-icon" aria-hidden="true">
              ✨
            </span>
            <div>
              <div className="brand-title">AI Career Navigator</div>
              <div className="brand-subtitle">Nâng cấp hành trình nghề nghiệp của bạn</div>
            </div>
          </div>
        </div>

        <div className="user-menu" ref={rootRef}>
          <button
            type="button"
            className="user-trigger"
            aria-haspopup="true"
            aria-expanded={open}
            onClick={onToggle}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onToggle();
              }
            }}
          >
            <div className="avatar" aria-hidden="true">
              {userName
                .split(" ")
                .map((p) => p[0])
                .slice(0, 2)
                .join("")}
            </div>
            <div className="user-labels">
              <span className="user-name">{userName}</span>
              <span className="user-role">{userSubtitle}</span>
            </div>
            <span className="user-status">{userStatus}</span>
          </button>

          <div className={`menu-panel ${open ? "open" : ""}`} role="menu" aria-label="Menu người dùng">
            <div className="menu-head">
              <div className="menu-head-title">Xin chào, {userName.split(" ")[0]}</div>
              <div className="menu-head-subtitle">
                {account ? "Lịch sử chat đang lưu trên hệ thống theo tài khoản của bạn." : "Đăng nhập để lưu lịch sử chat vào hệ thống."}
              </div>
            </div>
            <div className="menu-grid">
              {menuItems.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  className="menu-item"
                  onClick={() => {
                    item.onClick();
                    setOpen(false);
                  }}
                  role="menuitem"
                >
                  <span className="item-icon" aria-hidden="true">
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      {authOpen ? (
        <div className="auth-backdrop" onClick={() => setAuthOpen(false)}>
          <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
            <div className="auth-header">
              <h3>{authMode === "login" ? "Đăng nhập" : "Đăng ký tài khoản"}</h3>
              <button type="button" className="auth-close" onClick={() => setAuthOpen(false)}>
                ✕
              </button>
            </div>

            <div className="auth-tabs">
              <button
                type="button"
                className={`auth-tab ${authMode === "login" ? "active" : ""}`}
                onClick={() => setAuthMode("login")}
              >
                Đăng nhập
              </button>
              <button
                type="button"
                className={`auth-tab ${authMode === "register" ? "active" : ""}`}
                onClick={() => setAuthMode("register")}
              >
                Đăng ký
              </button>
            </div>

            <form className="auth-form" onSubmit={onSubmitAuth}>
              {authMode === "register" ? (
                <label>
                  Họ và tên
                  <input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Nguyễn Văn A" />
                </label>
              ) : null}

              <label>
                Email
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="ban@example.com" />
              </label>

              <label>
                Mật khẩu
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
              </label>

              {authError ? <div className="auth-error">{authError}</div> : null}

              <button className="btn auth-submit" type="submit" disabled={authLoading}>
                {authLoading ? "Đang xử lý..." : authMode === "login" ? "Đăng nhập" : "Tạo tài khoản"}
              </button>
            </form>
          </div>
        </div>
      ) : null}
    </>
  );
}
