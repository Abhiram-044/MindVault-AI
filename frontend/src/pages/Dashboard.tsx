import Sidebar from "@/components/layout/Sidebar";
import ChatWindow from "@/components/chat/ChatWindow";

export default function Dashboard() {
    return (
        <div className="flex h-screen bg-background">
            <Sidebar />

            <div className="flex-1 flex flex-col">
                <ChatWindow />
            </div>
        </div>
    )
}