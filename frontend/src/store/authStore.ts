import { create } from "zustand";
import { persist } from "zustand/middleware";
import { loginUser, getCurrentUser } from "@/api/services/authAPI";

interface User {
  id: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isHydrated: boolean;
  logoutTimer: NodeJS.Timeout | null; 
  expiresAt: number | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  startLogoutTimer: () => void;
  setHydrated: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isHydrated: false,
      logoutTimer: null,
      expiresAt: null,

      login: async (email, password) => {
        const res1 = await loginUser(email, password);
        const token = res1.access_token;
        const res2 = await getCurrentUser(token);

        const expiresAt = Date.now() + 60 * 60 * 1000;

        set({
          token,
          user: res2 ?? null,
          expiresAt,
        });
        
        get().startLogoutTimer();
      },

      logout: () => {
        const timer = get().logoutTimer;

        console.log(timer);

        if (timer) clearTimeout(timer);

        set({
          user: null,
          token: null,
          logoutTimer: null,
        });

        localStorage.removeItem("mindvault-chat");
      },

      startLogoutTimer: () => {
          const { expiresAt, logoutTimer} = get();

          if (!expiresAt) return;
          if (logoutTimer) clearTimeout(logoutTimer);

          const remainingTime = expiresAt - Date.now();

          if (remainingTime <= 0) {
            get().logout();
            return;
          }

          const timer = setTimeout(() => {
            console.log("Session expired.");
            get().logout();
          }, remainingTime);

          set({ logoutTimer: timer });
      },

      setHydrated: () => set({ isHydrated: true }),
    }),
    {
      name: "mindvault-chat",
      onRehydrateStorage: () => (state) => {
        state?.setHydrated();

        if (state?.token) {
          state.startLogoutTimer();
        }
      }
    }
  )
);