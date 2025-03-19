from phi.agent import Agent
from phi.knowledge.website import WebsiteKnowledgeBase
from phi.model.google import Gemini
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.tools.googlesearch import GoogleSearch
from phi.vectordb.pgvector import PgVector
from phi.embedder.google import GeminiEmbedder
import os
from dotenv import load_dotenv
load_dotenv()

# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector(
#         table_name="recipes",
#         db_url=os.getenv("DATABASE_URL"),
#         embedder=GeminiEmbedder(id="gemini-2.0-flash"),
#     ),
# )

knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.phidata.com/introduction"],
    # Number of links to follow from the seed URLs
    max_links=10,
    # Table name: ai.website_documents
    vector_db=PgVector(
        table_name="website_documents",
        db_url=os.getenv("DATABASE_URL"),
        embedder=GeminiEmbedder(id="gemini-2.0-flash"),
    ),
)

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    knowledge=knowledge_base,
    tools=[GoogleSearch()],
    description="You are a agent that helps users find the latest news and answers to their questions.",
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

agent.knowledge.load(recreate=False)
# agent.print_response("can you give me the small code snippets for integrate the ai agent using phidata?", stream=True)

def chat():
    print("🤖 AI Assistant: Hello! I'm here to help you. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("🤖 AI Assistant: Goodbye! Have a great day!")
            break
            
        if user_input:
            print("\n🤖 AI Assistant:", end=" ")
            agent.print_response(user_input, stream=True)

if __name__ == "__main__":
    agent.knowledge.load(recreate=False)
    chat()