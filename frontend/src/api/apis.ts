const BASE_URL = import.meta.env.VITE_BACKEND_URL;

// AUTH
export const authEndpoints = {
  REGISTER_API: `${BASE_URL}/auth/register`,
  LOGIN_API: `${BASE_URL}/auth/login`,
  ME_API: `${BASE_URL}/auth/me`,
};

export const chatEndpoints = {
  CREATE_SESSION: `${BASE_URL}/chat/session`,
  GET_SESSIONS: `${BASE_URL}/chat/sessions`,
  GET_MESSAGE: (id: string) => `${BASE_URL}/chat/${id}`,
  SEND_MESSAGE: `${BASE_URL}/chat/message/stream`,
  DELETE_SESSION: (id: string) => `${BASE_URL}/chat/session/${id}`
}

export const uploadEndpoints = {
  UPLOAD_DOC: `${BASE_URL}/files/upload`,
  GET_DOC_STATUS: (id: string) => `${BASE_URL}/files/${id}/status`
}