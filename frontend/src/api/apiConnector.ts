import axios from "axios";

export const axiosInstance = axios.create({
    withCredentials: true
});

export const apiConnector = (
    method: string,
    url: string,
    bodyData?: any,
    headers?: any,
    params?: any
) => {
    return axiosInstance({
        method,
        url,
        data: bodyData || null,
        headers: headers || null,
        params: params || null,
    });
};