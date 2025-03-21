const { uploadFileToS3 } = require('../services/s3Service');
const { deactivateOldFiles } = require('../services/fileService');
const File = require('../models/FileModel');

const uploadFile = async (req, res, next) => {
    try {
        const file = req.file;

        if (!file) {
            return res.status(400).json({ message: 'File is required.' });
        }

        // Validate file type
        if (!file.mimetype.includes('csv')) {
            return res.status(400).json({ message: 'Only CSV files are allowed.' });
        }

        // Deactivate all previous files
        await deactivateOldFiles();

        // Upload file to S3
        const result = await uploadFileToS3(file);

        // Save file metadata to MongoDB
        const newFile = new File({
            file_name: file.originalname,
            file_url: result.Location,
            file_type: file.mimetype,
            is_active: true
        });
        await newFile.save();

        res.status(200).json({ 
            message: 'File uploaded successfully!', 
            url: result.Location 
        });
    } catch (err) {
        next(err);
    }
};

module.exports = { uploadFile };