import ChatInput from "@/components/chat/ChatInput";
import EmptyState from "@/components/chat/EmptyState";
import MessageBubble from "@/components/chat/MessageBubble"
import { useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { useChatStore } from "@/store/chatStore";

export default function ChatWindow() {
    const { sessionId } = useParams();
    const { messages, loadMessages, isStreaming } =
        useChatStore();

    const bottomRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (sessionId) {
            loadMessages(sessionId);
        } else {
            
        }
    }, [sessionId]);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({
            behavior: "smooth"
        });
    }, [messages, isStreaming])

    return (
        <div className="flex flex-col h-full">

            <div className="flex-1 overflow-y-auto p-6">
                {messages.length === 0 ? (
                    <EmptyState />
                ) : messages.map((m, i) => (
                    <MessageBubble 
                        key={i}
                        role={m.role}
                        content={m.content}
                    />
                ))}
            </div>
            <div ref={bottomRef} />
            
            <ChatInput />
        </div>
    )
}