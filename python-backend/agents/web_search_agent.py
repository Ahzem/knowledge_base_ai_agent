from phi.agent import Agent
from phi.tools.googlesearch import GoogleSearch
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv

load_dotenv()

class WebSearchAgent:
    def __init__(self):
        self.agent = Agent(
            name="Web Search Agent",
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.7
            ),
            tools=[GoogleSearch()],
            description="Expert web search agent that finds relevant and up-to-date information",
            instructions=[
                "Always include sources and dates in responses",
                "Prioritize recent information",
                "Verify information from multiple sources when possible",
                "Format responses in clear sections",
                "Include relevant quotes when appropriate"
            ],
            add_history_to_messages=True,
            num_history_responses=5,
            show_tool_calls=True,
            markdown=True
        )

    def search(self, query):
        """
        Perform a web search and return formatted results
        """
        try:
            enhanced_query = f"""
            Search Request: {query}
            
            Please provide:
            1. Most relevant and recent information
            2. Sources for each piece of information
            3. Dates when the information was published
            4. A brief summary of key points
            """
            
            response = self.agent.run(enhanced_query)
            return response.content if response else "Unable to find relevant information."

        except Exception as e:
            print(f"Search Error: {str(e)}")
            return "Error: Unable to complete search due to technical issues."

# Example usage
if __name__ == "__main__":
    web_agent = WebSearchAgent()
    
    # Test queries
    test_queries = [
        "What are the latest developments in AI technology?",
        "Show me recent news about renewable energy",
        "What are the current trends in e-commerce?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        print(web_agent.search(query))