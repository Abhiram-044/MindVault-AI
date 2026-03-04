import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useEffect } from "react";
import { useChatStore } from "@/store/chatStore";
import UploadPanel from "@/components/upload/UploadPanel";
import { Trash2 } from "lucide-react";
import { LogOut } from "lucide-react";
import { useAuthStore } from "@/store/authStore";

export default function Sidebar() {
    const navigate = useNavigate()
    const { sessions, loadSessions, startNewChat, deleteChat, activeSession } =
        useChatStore();
    const logout = useAuthStore((s) => s.logout)

    useEffect(() => {
        loadSessions();
    }, [])

    const handleLogout = () => {
        logout();
        localStorage.clear();
        navigate("/login")
    }

    const handleNewChat = async () => {
        const id = await startNewChat();
        navigate(`/chat/${id}`);
    }

    const handleDelete = async (
        e: React.MouseEvent,
        sessionId: string
    ) => {
        e.stopPropagation();

        const confirmed = window.confirm(
            "Delete this chat?"
        );

        if (!confirmed) return;

        const newActive = await deleteChat(sessionId);

        navigate(`/chat/${newActive}`)
    }

    return (
        <div className="w-64 border-r h-screen flex flex-col">

            <div className="p-4">
                <Button className="w-full" onClick={handleNewChat}>
                    + New Chat
                </Button>
            </div>

            <div className="flex-1 overflow-y-auto px-4 space-y-2">
                {sessions.map((s: any) => {
                    const isActive = s._id === activeSession;

                    return (
                        <div
                            key={s._id}
                            onClick={() => navigate(`/chat/${s._id}`)}
                            className={`
                                    group
                                    flex
                                    items-center
                                    justify-between
                                    cursor-pointer
                                    p-2
                                    rounded
                                    transition

                                    ${isActive
                                    ? "bg-muted font-medium"
                                    : "hover:bg-muted"
                                }
                                `}
                        >
                            <span className="truncate text-sm">
                                {s.title}
                            </span>

                            <button
                                onClick={(e) => handleDelete(e, s._id)}
                                className={`
                                        transition
                                        hover:text-red-500
                                        ${isActive
                                        ? "opacity-100 text-muted-foreground"
                                        : "opacity-0 group-hover:opacity-100 text-muted-foreground"
                                    }
                                        `}
                                >
                                <Trash2 size={16} />
                            </button>
                        </div>
                    );
                })}
            </div>
            <div className="border-t p-4 space space-y-2">
                <UploadPanel />

                <Button
                    variant="outline"
                    className="w-full flex items-center gap-2 text-red-500 hover:text-red-600"
                    onClick={handleLogout}
                >
                    <LogOut size={16} />
                    Logout
                </Button>
            </div>
        </div>
    )
}