import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

interface Props {
    role: "user" | "assistant";
    content: string;
}

export default function MessageBubble({ role, content }: Props) {
    const isUSer = role == "user";

    return (
        <div
            className={`flex mb-4 ${isUSer ? "justify-end" : "justify-start"
                }`}
        >
            <div
                className={`max-w-xl px-4 py-3 rounded-lg ${isUSer
                        ? "bg-black text-white"
                        : "bg-muted"
                    }`}
            >
                {isUSer ? (
                    content
                ) : (
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                    >
                        {content}
                    </ReactMarkdown>
                )}
            </div>
        </div>
    )
}