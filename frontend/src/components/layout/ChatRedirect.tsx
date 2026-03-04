import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useChatStore } from "@/store/chatStore";

export default function ChatRedirect() {
    const navigate = useNavigate();

    const {
        sessions,
        loadSessions,
        startNewChat,
    } = useChatStore();

    useEffect(() => {
        const init = async () => {

            let currentSessions = sessions;

            if (currentSessions.length === 0) {
                await loadSessions();
                currentSessions =
                    useChatStore.getState().sessions;
            }

            if (currentSessions.length > 0) {
                navigate(
                    `/chat/${currentSessions[0]._id}`,
                    { replace: true }
                );
                return;
            }

            const id = await startNewChat();

            navigate(`/chat/${id}`, {
                replace: true,
            });
        };

        init();
    }, []);

    return null;
}