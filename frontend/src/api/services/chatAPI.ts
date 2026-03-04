import { apiConnector } from "../apiConnector";
import { chatEndpoints } from "../apis";
import toast from "react-hot-toast";

const {
    CREATE_SESSION,
    GET_SESSIONS,
    GET_MESSAGE,
    SEND_MESSAGE,
    DELETE_SESSION
} = chatEndpoints;

export async function createSession(token: string) {
    try {
        const res = await apiConnector(
            "POST",
            CREATE_SESSION,
            {},
            {
                Authorization: `Bearer ${token}`
            }
        );
        return res.data;
    } catch (error: any) {
        toast.error(
            error?.response?.data?.detail || "Cannot Create New Chat"
        )
    }
}

export async function getSessions(token: string) {
    try {
        const res = await apiConnector(
            "GET",
            GET_SESSIONS,
            null,
            {
                Authorization: `Bearer ${token}`
            }
        );
        return res.data;
    } catch (error: any) {
        toast.error(
            error?.response?.data?.detail || "Cannot Obtain Sessions. \nLogin Again."
        )
    }
}

export async function deleteSession(token: string, session_id: string) {
    try {
        const res = await apiConnector(
            "DELETE",
            DELETE_SESSION(session_id),
            null,
            {
                Authorization: `Bearer ${token}`
            }
        );
        return res.data;
    } catch (error: any) {
        toast.error(
            error?.response?.data?.detail || "Cannot Obtain Sessions. \nLogin Again."
        )
    }
}

export async function streamMessage(
    sessionId: string,
    message: string,
    token: string,
) {
    try {
        const response = await fetch(
            SEND_MESSAGE,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    query: message,
                }),
            }
        );

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(
                `Streaming failed: ${response.status} ${errorText}`
            );
        }

        if (!response.body) {
            throw new Error("ReadableStream not supported");
        }

        return response;

    } catch (error: any) {
        console.error("STREAM MESSAGE ERROR:", error);
        throw error;
    }
}

export async function getMessage(
    sessionId: string,
    token: string
) {
    try {
        const res = await apiConnector(
            "GET",
            GET_MESSAGE(sessionId),
            null,
            {
                Authorization: `Bearer ${token}`
            }
        );
        return res.data;
    } catch (error: any) {
        toast.error(
            error?.response?.data?.detail || "Cannot get chat. Login Again."
        )
    }
}