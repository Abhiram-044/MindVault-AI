import { useState } from "react";
import { registerUser } from "@/api/services/authAPI";
import { useNavigate } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { Link } from "react-router-dom";

export default function Register() {
    const navigate = useNavigate();

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleRegister = async () => {
        if (!email || !password) return;

        await registerUser(email, password, navigate)
    };

    return (
        <div className="flex h-screen items-center justify-center">
            <Toaster />
            <div className="w-96 space-y-4">
                <h1 className="text-2xl font-bold">Welcome to MindVault AI</h1>
                <h1 className="text-xl font-semibold">Your Personal AI Knowledge Base</h1>
                <h1 className="text-xl">Create account to get started.</h1>
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
                    onClick={handleRegister}
                    className="w-full bg-black text-white p-2"
                >
                    Register
                </button>
                <p className="text-sm text-center text-muted-foreground">
                    Already Have an account?{" "}
                    <Link
                        to="/login"
                        className="text-black font-medium hover:underline"
                    >
                        Login
                    </Link>
                </p>
            </div>
        </div>
    );
}