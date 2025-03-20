const mongoose = require("mongoose");
require("dotenv").config();

const connectDB = async () => {
  try {
    const connectionOptions = {
      authSource: "admin",
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
    };

    await mongoose.connect(process.env.MONGO_URI, connectionOptions);
    console.log("✅ MongoDB connected successfully");
  } catch (err) {
    console.error("❌ MongoDB connection error:");
    if (err.name === "MongoServerError" && err.code === 8000) {
      console.error("Authentication failed - Please check your credentials");
    } else if (err.name === "MongoTimeoutError") {
      console.error(
        "Connection timeout - Please check your connection string and network"
      );
    } else {
      console.error(err.message);
    }
    process.exit(1);
  }
};

module.exports = connectDB;
