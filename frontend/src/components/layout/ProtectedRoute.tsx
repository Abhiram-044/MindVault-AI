import { Navigate } from "react-router-dom";
import { useAuthStore } from "@/store/authStore";

export default function ProtectedRoute({
    children,
}: {
    children: React.ReactNode;
}) {
    const { token, isHydrated } = useAuthStore();

    if (!isHydrated) {
        return (
            <div className="h-screen flex items-center justify-center">
                Loading...
            </div>
        );
    }

    if (!token) {
        return <Navigate to="/login" replace />;
    }

    return children;
}