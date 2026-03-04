import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Login from '@/pages/Login';
import Register from '@/pages/Register';
import Dashboard from '@/pages/Dashboard';
import ProtectedRoute from '@/components/layout/ProtectedRoute';
import ChatRedirect from '@/components/layout/ChatRedirect';

const AppRoutes = () => {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route
                    path="/chat"
                    element={
                        <ProtectedRoute>
                            <ChatRedirect />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/chat/:sessionId"
                    element={
                        <ProtectedRoute>
                            <Dashboard />
                        </ProtectedRoute>
                    }
                />
            </Routes>
        </BrowserRouter>
    );
};

export default AppRoutes;