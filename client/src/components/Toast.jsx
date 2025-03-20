import React, { useEffect } from 'react';
import '../styles/Toast.css';
import { FaClipboard, FaCheck, FaTimes } from 'react-icons/fa';

const Toast = ({ message, url, onClose, isVisible }) => {
    useEffect(() => {
        if (isVisible) {
            const timer = setTimeout(() => {
                onClose();
            }, 5000);
            return () => clearTimeout(timer);
        }
    }, [isVisible, onClose]);

    const [copied, setCopied] = React.useState(false);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={`toast ${isVisible ? 'toast-visible' : ''}`}>
            <div className="toast-content">
                <div className="toast-icon">✨</div>
                <div className="toast-message">
                    <p>{message}</p>
                    <div className="url-container">
                        <span className="url-text">{url}</span>
                        <button className="copy-button" onClick={handleCopy}>
                            {copied ? 
                                <FaCheck className="icon" /> : 
                                <FaClipboard className="icon" />
                            }
                        </button>
                    </div>
                </div>
                <button className="toast-close" onClick={onClose}><FaTimes /></button>
            </div>
        </div>
    );
};

export default Toast;