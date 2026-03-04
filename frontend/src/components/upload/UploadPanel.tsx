import { useUploadStore } from "@/store/uploadStore";
import { useRef, useEffect } from "react";

export default function UploadPanel() {

  const fileRef = useRef<HTMLInputElement>(null);

  const {
    uploadFile,
    uploading,
    processing,
  } = useUploadStore();

  const handleFile = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!e.target.files?.[0]) return;

    await uploadFile(e.target.files[0]);
  };

  
  useEffect(() => {
    if (!uploading && !processing && fileRef.current) {
      fileRef.current.value = "";
    }
  }, [uploading, processing]);

  return (
    <div className="border-b p-4 space-y-3">

      <label className="block cursor-pointer">
        <div className="border rounded-lg p-3 text-center hover:bg-muted">
          Upload PDF, TXT
        </div>

        <input
          ref={fileRef}
          type="file"
          accept=".pdf, .txt"
          onChange={handleFile}
          className="hidden"
        />
      </label>

      {uploading && (
        <p className="text-xs">
          Uploading...
        </p>
      )}

      {processing && (
        <p className="text-xs text-blue-500">
          Processing document...
        </p>
      )}

    </div>
  );
}