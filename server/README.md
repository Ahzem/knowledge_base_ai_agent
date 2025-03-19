# Knowledge Base AI Agent

An intelligent AI assistant built with phidata that can search through documentation, answer questions, and provide real-time information using Google Search integration.

## Features

- Website knowledge base integration
- Google Search capability
- Real-time streaming responses
- Interactive chat interface
- Markdown formatting support
- Vector database storage using pgvector

## Prerequisites

- Python 3.8+
- PostgreSQL database with pgvector extension
- Google API key
- Environment variables setup

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd knowledge_base_ai_agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
DATABASE_URL=your_postgresql_connection_string
GOOGLE_API_KEY=your_google_api_key
```

## Usage

Run the application:
```bash
python app.py
```

The chat interface will start, allowing you to:
- Ask questions about the loaded documentation
- Search for real-time information
- Exit the chat using 'exit', 'quit', or 'bye'

## Configuration

The agent is configured with:
- Gemini model for text generation
- Website knowledge base (currently set to phidata docs)
- PostgreSQL vector database for storing embeddings
- Google Search integration

You can modify the knowledge base by updating the URLs in `app.py`:

```python
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.phidata.com/introduction"],
    max_links=10,
    vector_db=PgVector(...)
)
```

## Dependencies

Key dependencies include:
- phidata
- google-generativeai
- pgvector
- psycopg2
- python-dotenv

For a complete list, see `requirements.txt`.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]