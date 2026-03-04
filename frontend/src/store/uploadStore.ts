import { create } from "zustand";
import {
  uploadDocument,
  getDocStatus,
} from "@/api/services/documentAPI";
import { useAuthStore } from "./authStore";

interface UploadState {
  uploading: boolean;
  processing: boolean;
  progress: number;
  fileId: string | null;

  uploadFile: (file: File) => Promise<void>;
  resetUpload: () => void;
}

export const useUploadStore = create<UploadState>((set, get) => ({

  uploading: false,
  processing: false,
  progress: 0,
  fileId: null,

  resetUpload: () =>
    set({
      uploading: false,
      processing: false,
      progress: 0,
      fileId: null,
    }),

  uploadFile: async (file) => {
    try {
      const token = useAuthStore.getState().token!;

      set({
        uploading: true,
        progress: 20,
      });

      
      const res = await uploadDocument(file, token);

      const file_id = res.data.file_id;

      set({
        uploading: false,
        processing: true,
        progress: 100,
        fileId: file_id,
      });

      
      const pollStatus = async () => {
        const statusRes = await getDocStatus(
          token,
          file_id
        );

        const status = statusRes.status;

        if (status === "processed") {
          set({ processing: false });

          
          setTimeout(() => {
            get().resetUpload();
          }, 10000);

          return;
        }

        if (status === "failed") {
          get().resetUpload();
          return;
        }

        
        setTimeout(pollStatus, 2000);
      };

      setTimeout(pollStatus, 2000)

    } catch (err) {
      console.error(err);
      get().resetUpload();
    }
  },
}));