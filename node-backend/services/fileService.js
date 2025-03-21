const File = require('../models/FileModel');

/**
 * Deactivate all previously active files
 */
const deactivateOldFiles = async () => {
    try {
        await File.updateMany(
            { is_active: true },
            { $set: { is_active: false } }
        );
    } catch (error) {
        console.error('Error deactivating old files:', error);
        throw error;
    }
};

/**
 * Get URLs of active files from MongoDB
 */
const getFileUrls = async () => {
    try {
        const files = await File.find({ is_active: true }, { file_url: 1, _id: 0 });
        return files.map(file => file.file_url);
    } catch (error) {
        console.error('Error fetching file URLs:', error);
        throw error;
    }
};

module.exports = { deactivateOldFiles, getFileUrls };