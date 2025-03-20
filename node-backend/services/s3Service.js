const { PutObjectCommand } = require('@aws-sdk/client-s3');
const s3Client = require('../config/aws');

const uploadFileToS3 = async (file) => {
    const params = {
        Bucket: process.env.AWS_S3_BUCKET_NAME,
        Key: file.originalname,
        Body: file.buffer,
        ContentType: file.mimetype,
    };

    const command = new PutObjectCommand(params);
    const response = await s3Client.send(command);
    
    return {
        Location: `https://${params.Bucket}.s3.${process.env.AWS_REGION}.amazonaws.com/${params.Key}`,
        ...response
    };
};

module.exports = { uploadFileToS3 };