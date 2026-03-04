import { authEndpoints } from "../apis";
import { apiConnector } from "../apiConnector";
import toast from "react-hot-toast";

const {
    REGISTER_API,
    LOGIN_API,
    ME_API
} = authEndpoints;

export async function registerUser(
    email: string,
    password: string,
    navigate: any
) {
    try {
        const response = await apiConnector(
            "POST",
            REGISTER_API,
            {
                "email": email,
                "password": password
            }
        );

        toast.success("Account Created Successfully");
        navigate("/login");
        return response.data;
    } catch (error: any) {
        toast.error(
            error?.response?.data?.detail || "Registration failed"
        );
        throw error;
    }
}

export async function loginUser(
    email: string,
    password: string
) {
    try {
        const formData = new URLSearchParams();
        formData.append("username", email);
        formData.append("password", password);

        const response = await apiConnector(
            "POST",
            LOGIN_API,
            formData,
            {
                "Content-Type":
                    "application/x-www-form-urlencoded",
            }
        );

        return response.data;

    } catch (error: any) {
        toast.error(
            error?.response?.data?.detail || "Login Failed"
        );
        throw error;
    }
}

export async function getCurrentUser(token: string) {
    const response = await apiConnector(
        "GET",
        ME_API,
        null,
        {
            Authorization: `Bearer ${token}`,
        }
    );

    return response.data;
}