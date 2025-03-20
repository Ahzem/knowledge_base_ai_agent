const { uploadFileToS3 } = require('../services/s3Service');
const File = require('../models/FileModel');

const uploadFile = async (req, res, next) => {
    try {
        const file = req.file;
        // const { session_id } = req.body; // Session ID from Python AI agent

        if (!file) {
            return res.status(400).json({ message: 'File is required.' });
        }
        
        // if (!file || !session_id) {
        //     return res.status(400).json({ message: 'File and session ID are required.' });
        // }

        // Upload file to S3
        const result = await uploadFileToS3(file);

        // Save file metadata to MongoDB
        const newFile = new File({
            // session_id,
            file_url: result.Location,
        });
        await newFile.save();

        res.status(200).json({ message: 'File uploaded successfully!', url: result.Location });
    } catch (err) {
        next(err);
    }
};

module.exports = { uploadFile };