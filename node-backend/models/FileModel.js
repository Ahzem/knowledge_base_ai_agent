const mongoose = require('mongoose');

const fileSchema = new mongoose.Schema({
    session_id: { type: String },
    file_url: { type: String, required: true },
    uploaded_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model('File', fileSchema);