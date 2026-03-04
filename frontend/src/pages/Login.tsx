import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/store/authStore";
import { Toaster } from "react-hot-toast";
import { useChatStore } from "@/store/chatStore";
import { Link } from "react-router-dom";

export default function Login() {
    const login = useAuthStore((s) => s.login)
    const navigate = useNavigate()

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleLogin = async () => {
        await login(email, password);
        const chatStore = useChatStore.getState();

        await chatStore.loadSessions();

        let sessionId;

        if (chatStore.sessions.length > 0) {
            sessionId = chatStore.sessions[0]._id;
        } else {
            sessionId = await chatStore.startNewChat()
        }
        navigate(`/chat/${sessionId}`)
    };

    useEffect(() => (
        console.log(useChatStore.getState().sessions)
    ), [])

    return (
        <div className="flex h-screen items-center justify-center">
            <Toaster />
            <div className="w-96 space-y-4">
                <h1 className="text-2xl font-bold">Welcome Back to MindVault AI</h1>
                <h1 className="text-xl font-bold">Login to Continue</h1>
                <input
                    className="w-full border p-2"
                    placeholder="Email"
                    onChange={(e) => setEmail(e.target.value)}
                />

                <input
                    type="password"
                    className="w-full border p-2"
                    placeholder="Password"
                    onChange={(e) => setPassword(e.target.value)}
                />

                <button
                    onClick={handleLogin}
                    className="w-full bg-black text-white p-2"
                >
                    Login
                </button>
                <p className="text-sm text-center text-muted-foreground">
                    Don't have an account?{" "}
                    <Link
                        to="/register"
                        className="text-black font-medium hover:underline"
                    >
                        Register
                    </Link>
                </p>
            </div>
        </div>
    )
}