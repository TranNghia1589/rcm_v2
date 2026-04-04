export type AccountInfo = {
  userId: number;
  fullName: string;
  email: string;
  token: string;
};

const ACCOUNT_STORAGE_KEY = "rcm_user_account";
export const ACCOUNT_CHANGED_EVENT = "rcm-account-changed";

export function getAccount(): AccountInfo | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(ACCOUNT_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as AccountInfo;
    if (!parsed?.email || !parsed?.fullName || !parsed?.token || !parsed?.userId) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function setAccount(account: AccountInfo): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(ACCOUNT_STORAGE_KEY, JSON.stringify(account));
  window.dispatchEvent(new Event(ACCOUNT_CHANGED_EVENT));
}

export function clearAccount(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(ACCOUNT_STORAGE_KEY);
  window.dispatchEvent(new Event(ACCOUNT_CHANGED_EVENT));
}
