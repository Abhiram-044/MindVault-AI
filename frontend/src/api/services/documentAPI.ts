import { apiConnector } from "../apiConnector";
import { uploadEndpoints } from "../apis";

const {
    UPLOAD_DOC,
    GET_DOC_STATUS
} = uploadEndpoints;

export const uploadDocument = async (
  file: File,
  token: string
) => {

  const formData = new FormData();
  formData.append("file", file);

  return apiConnector(
    "POST",
    UPLOAD_DOC,
    formData,
    {
      Authorization: `Bearer ${token}`,
      "Content-Type": "multipart/form-data",
    }
  );
};

export async function getDocStatus(token: string, file_id: string) {
    try {
        const res = await apiConnector(
            "GET",
            GET_DOC_STATUS(file_id),
            null,
            {
                Authorization: `Bearer ${token}`
            }
        );
        return res.data;
    } catch (error: any) {
        console.error("Cannot obtain status", error);
        throw error;
    }
}