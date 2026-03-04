import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useChatStore } from "@/store/chatStore";

export default function ChatInput() {

    const [message, setMessage] = useState("");
    const sendMessage = useChatStore(
        s => s.sendMessage
    );
    
    const handleSend = async () => {
        if (!message.trim()) return;

        await sendMessage(message);
        setMessage("");
    };

    return (
        <div className="border-t p-4 flex gap-2">

            <input
                className="flex-1 border rounded p-2"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask MindVault..."
            />

            <Button onClick={handleSend}>
                Send
            </Button>

        </div>
    )
}