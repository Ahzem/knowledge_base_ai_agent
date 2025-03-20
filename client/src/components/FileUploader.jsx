import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import useFileUpload from '../hooks/useFileUpload';
import { FaCloudUploadAlt, FaFolder, FaFolderOpen } from 'react-icons/fa';
import Toast from '../components/Toast';
import '../styles/FileUploader.css';

const FileUploader = () => {
    const { uploadFile, isUploading, error } = useFileUpload();
    const [selectedFile, setSelectedFile] = useState(null);
    const [toast, setToast] = useState({ visible: false, message: '', url: '' });

    const onDrop = async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (file) {
            setSelectedFile(file);
            try {
                const result = await uploadFile(file);
                setToast({
                    visible: true,
                    message: 'File uploaded successfully! 🎉',
                    url: result.url
                });
                setSelectedFile(null);
            } catch (err) {
                console.error(err);
            }
        }
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
        onDrop,
        multiple: false
    });

    return (
        <>
            <div 
                {...getRootProps()} 
                className={`dropzone ${isUploading ? 'uploading' : ''}`}
            >
                <input {...getInputProps()} />
                <div className="dropzone-content">
                    <div className="upload-icon">
                        {isUploading ? <FaCloudUploadAlt /> : isDragActive ? <FaFolderOpen /> : <FaFolder />}
                    </div>
                    
                    {isUploading ? (
                        <p className="upload-text">Uploading your file...</p>
                    ) : isDragActive ? (
                        <p className="upload-text">Drop your file here!</p>
                    ) : (
                        <p className="upload-text">
                            Drag & drop a file here, or click to select
                        </p>
                    )}

                    {selectedFile && !isUploading && (
                        <div className="file-preview">
                            Selected: {selectedFile.name}
                        </div>
                    )}

                    {error && <p className="error-message">{error}</p>}
                </div>
            </div>
            <Toast 
                message={toast.message}
                url={toast.url}
                isVisible={toast.visible}
                onClose={() => setToast(prev => ({ ...prev, visible: false }))}
            />
        </>
    );
};

export default FileUploader;