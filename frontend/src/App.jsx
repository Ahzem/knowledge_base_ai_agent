import React from 'react';
import FileUploader from './components/FileUploader';
import './App.css';

const App = () => {
    return (
        <div className="container">
            <h1>File Upload</h1>
            <p className="subtitle">Upload your files securely to the cloud</p>
            <FileUploader />
        </div>
    );
};

export default App;