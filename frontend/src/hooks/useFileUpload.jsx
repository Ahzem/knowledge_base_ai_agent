import { useState } from 'react';
import api from '../utils/api';

const useFileUpload = () => {
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState(null);

    const uploadFile = async (file) => {
        setIsUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await api.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data;
        } catch (err) {
            setError(err.response?.data?.message || 'Error uploading file');
            throw err;
        } finally {
            setIsUploading(false);
        }
    };

    return { uploadFile, isUploading, error };
};

export default useFileUpload;