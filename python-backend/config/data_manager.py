import requests
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.api_url = "http://localhost:5000/api/urls"
        self.urls = []

    def fetch_urls_from_backend(self):
        """Fetch CSV URLs from backend API and return success status"""
        try:
            logger.info(f"Fetching URLs from: {self.api_url}")
            response = requests.get(self.api_url)
            
            if response.status_code == 200:
                self.urls = response.json().get('urls', [])
                logger.info(f"Successfully fetched {len(self.urls)} URLs: {self.urls}")
                return True
            else:
                logger.error(f"API response: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False

    def fetch_and_update_data(self):
        """Fetch data URLs and check if successful"""
        success = self.fetch_urls_from_backend()
        if success:
            logger.info("URL fetching successful")
        else:
            logger.warning("URL fetching failed")
        return success

    def get_urls(self):
        """Return the currently fetched URLs"""
        return self.urls

# Singleton instance
data_manager = DataManager()