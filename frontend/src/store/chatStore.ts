import { create } from "zustand";
import {
    createSession,
    getSessions,
    getMessage,
    streamMessage,
    deleteSession
} from "@/api/services/chatAPI";
import { useAuthStore } from "./authStore";
import { persist } from "zustand/middleware";

interface Message {
    role: "user" | "assistant";
    content: string;
}

interface ChatState {
    sessions: any[];
    messages: Message[];
    activeSession: string | null;
    isStreaming: boolean;

    loadSessions: () => Promise<void>;
    startNewChat: () => Promise<string>;
    deleteChat: (sessionId: string) => Promise<string>;
    loadMessages: (sessionId: string) => Promise<void>;
    sendMessage: (message: string) => Promise<void>;
}

export const useChatStore = create<ChatState>()(
    persist(
        (set, get) => ({
            sessions: [],
            messages: [],
            activeSession: null,
            isStreaming: false,

            loadSessions: async () => {
                const token = useAuthStore.getState().token!;
                const data = await getSessions(token);

                set({ sessions: data });
            },

            startNewChat: async () => {
                const token = useAuthStore.getState().token!;
                const session = await createSession(token);

                set(state => ({
                        sessions: [
                            session,
                            ...state.sessions
                        ],
                        activeSession: session._id,
                        messages: []
                    }));

                return session._id;
            },

            deleteChat: async (sessionId) => {
                try {
                    const token = useAuthStore.getState().token!;
                    await deleteSession(token, sessionId);

                    const state = get();

                    const updatedSessions = state.sessions.filter(
                        (s) => s._id !== sessionId
                    );

                    if (updatedSessions.length === 0) {
                        const session = await createSession(token);
                        set({
                            sessions: [session],
                            activeSession: session._id,
                            messages: [],
                        })
                        return session._id;
                    } 
                    const isActiveDeleted = state.activeSession === sessionId;
                    let updatedActiveSession = state.activeSession;
                    let messages = state.messages;
                    if (isActiveDeleted) {
                        updatedActiveSession = updatedSessions[0]._id;
                        messages = await getMessage(updatedActiveSession!, token);
                    }
                    
                    set({
                        sessions: updatedSessions,
                        activeSession: updatedActiveSession,
                        messages: messages
                    })
                    return updatedActiveSession!;

                } catch (error) {
                    console.error("Delete session failed:", error);
                    return get().activeSession!;
                }
            },

            loadMessages: async (sessionId) => {
                const token = useAuthStore.getState().token!;
                const msgs = await getMessage(sessionId, token);

                
                set({
                    messages: msgs,
                    activeSession: sessionId,
                });
            },

            sendMessage: async (message) => {
                try {
                    const token = useAuthStore.getState().token!;
                    const sessionId = get().activeSession!;

                    if (get().isStreaming) return;

                    set(state => ({
                        messages: [
                            ...state.messages,
                            { role: "user", content: message },
                            { role: "assistant", content: "" }
                        ],
                        isStreaming: true
                    }));

                    const response = await streamMessage(
                        sessionId,
                        message,
                        token
                    );

                    const reader = response.body!.getReader();
                    const decoder = new TextDecoder();

                    let buffer = "";

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });

                        const lines = buffer.split("\n");
                        buffer = lines.pop() || "";

                        for (const line of lines) {
                            if (!line.startsWith("data: ")) continue;
                            const chunk = line.replace("data: [NONE]", "");
                            const chunk2 = chunk.replace("data: ", "");

                            set(state => {
                                const updated = [...state.messages]
                                updated[updated.length - 1].content += chunk2;
                                return { messages: updated }
                            });
                        }
                    }

                } catch (error) {
                    console.error("Chat streaming failed:", error);

                    set(state => {
                        const updated = [...state.messages];
                        updated[updated.length - 1].content =
                            "Failed to generate response.";
                        return { messages: updated };
                    });
                } finally {
                    set({ isStreaming: false });
                }
            }

        }),
        {
            name: "mindvault-auth",
            partialize: (state) => ({
                sessions: state.sessions,
                activeSession: state.activeSession
            })
        }
    )
);