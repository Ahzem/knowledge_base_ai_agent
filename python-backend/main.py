from phi.agent import Agent
from agents.web_search_agent import WebSearchAgent
from agents.sales_analytics_agent import SalesFinanceAgent
from phi.knowledge.csv import CSVUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from agents.recommendation_agent import RecommendationAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_generation_agent import ReportGenerationAgent
from config.api import fetch_urls, get_csv_url
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class MainChatAgent:
    def __init__(self):
        # Fetch URLs using api.py instead of data_manager
        csv_url = get_csv_url()
        
        # Set up knowledge base with the CSV file - using urls (plural) parameter
        self.knowledge_base = CSVUrlKnowledgeBase(
            urls=[csv_url],  # Use urls parameter with a list
            vector_db=PgVector(
                table_name="sales_finance",
                db_url=os.getenv("DATABASE_URL"),
            ),
        )
        
        # Initialize individual specialized agents
        self.web_search_agent = Agent(
            name="Web Search Agent",
            role="Searches the web for information and current events",
            agent=WebSearchAgent(),
        )
        
        # Rest of your initialization code remains the same
        # ...
        
        # self.data_summary_agent = Agent(
        #     name="Data Summary Agent",
        #     role="Provides summaries and overviews of business data",
        #     agent=DataSummaryAgent(),
        # )
        
        self.financial_analytics_agent = Agent(
            name="Financial Analytics Agent",
            role="Analyzes financial metrics and provides business insights",
            agent=SalesFinanceAgent(),
        )
        
        # self.recommendation_agent = Agent(
        #     name="Recommendation Agent",
        #     role="Provides business recommendations based on data analysis",
        #     agent=RecommendationAgent(),
        # )
        
        # self.visualization_agent = Agent(
        #     name="Visualization Agent",
        #     role="Creates data visualizations and charts",
        #     agent=VisualizationAgent(),
        # )
        
        # self.report_generation_agent = Agent(
        #     name="Report Generation Agent",
        #     role="Creates comprehensive business reports",
        #     agent=ReportGenerationAgent(),
        # )
        
        # Create the team agent with all specialized agents
        self.team_agent = Agent(
            name="Business Intelligence Assistant",
            team=[
                self.web_search_agent,
                # self.data_summary_agent,
                self.financial_analytics_agent,
                # self.recommendation_agent,
                # self.visualization_agent,
                # self.report_generation_agent,
            ],
            instructions=[
                # "For simple greetings or casual questions, respond directly without using specialized agents.",
                # "For financial analysis queries (including revenue, profits, sales metrics), use the Financial Analytics Agent.",
                # "For product analysis queries (top-selling products, profitability, product performance), use the Financial Analytics Agent.",
                # # "For data overview requests or KPI summaries, use the Data Summary Agent.",
                # # "For recommendation requests (what should we do, suggest strategies), use the Recommendation Agent.",
                # # "For visualization requests (charts, graphs, plots), use the Visualization Agent.",
                # # "For report generation requests, use the Report Generation Agent.",
                # "For general information or current events, use the Web Search Agent.",
                # "If a query requires multiple agents, coordinate their responses into a coherent answer.",
                # "Always provide clear, actionable insights based on available data."
                "For simple greetings or casual questions, respond directly without using specialized agents.",
                "For ALL financial analysis queries (including revenue, profits, sales metrics), use the Financial Analytics Agent.",
                "For ALL product analysis queries (top-selling products, profitability, product performance), use the Financial Analytics Agent.",
                "For ALL sales data queries, use the Financial Analytics Agent.",
                "For general information or current events, use the Web Search Agent.",
                "If a query requires multiple agents, coordinate their responses into a coherent answer.",
                "Always provide clear, actionable insights based on available data."
            ],
            knowledge_base=self.knowledge_base,
            add_history_to_messages=True,
            num_history_responses=3,
            show_tool_calls=True,
            markdown=True,
        )

    # @staticmethod
    # def get_system_prompt():
    #     return """You are an expert AI assistant that coordinates multiple specialized agents:

    #     Available Agents:
    #     1. Web Search Agent: For internet research and current information
    #     2. Data Summary Agent: For business data summaries and KPI overviews
    #     3. Financial Analytics Agent: For financial data analysis and metrics
    #     4. Recommendation Agent: For business recommendations and insights
    #     5. Visualization Agent: For creating charts and graphs
    #     6. Report Generation Agent: For creating comprehensive reports

    #     For each user query:
    #     1. Determine the most appropriate agent(s)
    #     2. Route the query accordingly
    #     3. Combine responses if multiple agents are used
    #     4. Maintain conversation context
    #     5. Save important information for future reference
    #     """
        
    @staticmethod
    def get_system_prompt():
        return """You are an expert AI assistant that coordinates multiple specialized agents:

        Available Agents:
        1. Web Search Agent: For internet research and current information
        2. Financial Analytics Agent: For financial data analysis and metrics

        For each user query:
        1. Determine the most appropriate agent(s)
        2. Route the query accordingly
        3. Combine responses if multiple agents are used
        4. Maintain conversation context
        5. Save important information for future reference
        """

    # def process_query(self, query, context=None):
    #     """Process user query using the team agent"""
    #     try:
    #         # Check if it's a simple greeting or casual question
    #         simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    #         if query.lower() in simple_greetings:
    #             return f"Hi! How can I help you today?"
            
    #         # Check if it's a report generation request
    #         if "generate" in query.lower() and "report" in query.lower():
    #             # Extract month and year
    #             months = ["january", "february", "march", "april", "may", "june", 
    #                      "july", "august", "september", "october", "november", "december"]
                
    #             month_num = None
    #             year_num = None
                
    #             # Extract year (assuming 4-digit format)
    #             years = re.findall(r'\b(20\d\d)\b', query)
    #             if years:
    #                 year_num = int(years[0])
                
    #             # Extract month
    #             for i, month_name in enumerate(months, 1):
    #                 if month_name in query.lower():
    #                     month_num = i
    #                     break
                
    #             if month_num and year_num:
    #                 return self.generate_monthly_report(year_num, month_num)
    #             else:
    #                 return "I couldn't determine which month and year you want a report for. Please specify both clearly."
            
    #         # Use the team agent to process the query
    #         enhanced_query = f"""
    #         User Query: {query}
            
    #         Additional Context: {context if context else 'No additional context provided'}
            
    #         Current Date: {datetime.now().strftime('%Y-%m-%d')}
    #         """
            
    #         response = self.team_agent.run(enhanced_query)
    #         return response.content
    
    #     except Exception as e:
    #         error_msg = f"Error processing query: {str(e)}"
    #         print(error_msg)
    #         return error_msg

    # def generate_monthly_report(self, year, month, title=None):
    #     """Generate a monthly report for a specific year and month"""
    #     try:
    #         # Create a default title if none provided
    #         if not title:
    #             month_name = calendar.month_name[month]
    #             title = f"{month_name} {year} Business Performance Report"
            
    #         print(f"Generating report for {calendar.month_name[month]} {year}...")
            
    #         # Access the ReportGenerationAgent
    #         report_agent = self.report_generation_agent.agent
            
    #         # Ensure agent and generate_report method exist
    #         if not hasattr(report_agent, 'generate_report'):
    #             return "Error: Report generation functionality is not available in the agent."
            
    #         # Generate the report
    #         report_path = report_agent.generate_report(
    #             report_type="monthly",
    #             period=(year, month),
    #             title=title
    #         )
            
    #         # Verify report was created
    #         if os.path.exists(report_path):
    #             absolute_path = os.path.abspath(report_path)
    #             return f"Report generated successfully!\n\nFile: {absolute_path}"
    #         else:
    #             return f"Report generation process completed, but the file was not found at: {report_path}"
        
    #     except AttributeError as e:
    #         error_msg = f"Error: The report generation agent does not have the required methods: {str(e)}"
    #         print(error_msg)
    #         return error_msg
    #     except Exception as e:
    #         error_msg = f"Error generating report: {str(e)}"
    #         print(error_msg)
    #         return error_msg

    def process_query(self, query, context=None):
        """Process user query using the team agent"""
        try:
            # Check if it's a simple greeting or casual question
            simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
            if query.lower() in simple_greetings:
                return f"Hi! How can I help you today?"
            
            # Use the team agent to process the query
            enhanced_query = f"""
            User Query: {query}
            
            Additional Context: {context if context else 'No additional context provided'}
            
            Current Date: {datetime.now().strftime('%Y-%m-%d')}
            """
            
            response = self.team_agent.run(enhanced_query)
            return response.content
    
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg

def chat():
    """Interactive chat function"""
    agent = MainChatAgent()
    print("🤖 AI Assistant: Hello! I'm here to help you. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("🤖 AI Assistant: Goodbye! Have a great day!")
            break
            
        if user_input:
            print("\n🤖 AI Assistant:", end=" ")
            response = agent.process_query(user_input)
            print(response)
            
def initialize_system():
    """Initialize the system and update data"""
    print("Initializing AI assistant system...")
    
    # Update shared knowledge base using api.py
    print("Loading knowledge base...")
    urls = fetch_urls()
    
    if not urls or len(urls) == 0:
        print("Warning: Using fallback data due to knowledge base update failure")
    else:
        print(f"Knowledge base successfully loaded with {len(urls)} URLs")
    
    # Initialize main chat agent
    print("Initializing chat agent...")
    return MainChatAgent()

if __name__ == "__main__":
    agent = initialize_system()
    chat()