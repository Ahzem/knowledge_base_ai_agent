from phi.agent import Agent
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.model.openai import OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()

class SalesAnalystAgent:
    def __init__(self):
        # Initialize knowledge base
        self.knowledge_base = CSVKnowledgeBase(
            path="data/FanBudget.csv",
            vector_db=PgVector(
                table_name="sales_data",
                db_url=os.getenv("DATABASE_URL"),
            ),
        )
        
        # Initialize agent
        self.agent = Agent(
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            knowledge=self.knowledge_base,
            search_knowledge=True,
        )
        
    def get_insights(self, data_summary, month, year):
        """Generate AI insights from sales data"""
        try:
            prompt = f"""As a retail business analyst, analyze this sales data for {month}/{year}:

            # Overall Performance
            - Revenue, profit margins, transaction patterns
            - Impact of discounts and refunds

            # Product Performance
            - Top/underperforming products
            - Category performance
            - Inventory turnover

            # Customer Insights
            - Purchase patterns and segments
            - Customer retention
            - Response to promotions

            # Strategic Recommendations
            - Inventory optimization
            - Pricing strategy
            - Customer engagement

            Data: {data_summary}

            Provide a clear, professional analysis with specific metrics and actionable recommendations."""

            response = self.agent.run(prompt)
            return response.content if response else "Unable to generate insights."

        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return "Error: Unable to generate insights due to technical issues."