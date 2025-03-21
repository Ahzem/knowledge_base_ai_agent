const express = require("express");
const cors = require("cors");
const http = require("http");
const connectDB = require("./config/db");
const fileRoutes = require("./routes/fileRoutes");
const errorHandler = require("./middleware/errorHandler");
const logger = require("./utils/logger");

const app = express();
let server;

// Try multiple ports if the default is busy
const startServer = (portToTry) => {
  server = http.createServer(app);
  
  server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
      logger.warn(`Port ${portToTry} is already in use, trying ${portToTry + 1}...`);
      setTimeout(() => {
        server.close();
        startServer(portToTry + 1);
      }, 1000);
    } else {
      logger.error('Server error:', error);
    }
  });
  
  server.listen(portToTry, () => {
    logger.info(`Server running on http://localhost:${portToTry}`);
  });
};

// Initial setup
const init = async () => {
  try {
    // Connect to MongoDB
    await connectDB();

    // Middleware
    app.use(cors());
    app.use(express.json());

    // Routes
    app.use("/api", fileRoutes);
    
    // Default route
    app.get("/", (req, res) => {
      res.send("Data Analyst AI Agent API is running");
    });

    // Error handler
    app.use(errorHandler);

    // Start server
    const initialPort = process.env.PORT || 5000;
    startServer(initialPort);
    
  } catch (err) {
    logger.error("Failed to initialize server:", err);
    process.exit(1);
  }
};

// Handle graceful shutdown
const gracefulShutdown = () => {
  logger.info("Shutting down gracefully...");
  if (server) {
    server.close(() => {
      logger.info("Server closed");
      process.exit(0);
    });
    
    // Force close after 10 seconds
    setTimeout(() => {
      logger.error("Could not close connections in time, forcing shutdown");
      process.exit(1);
    }, 10000);
  }
};

// Listen for termination signals
process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);

// Start the application
init();