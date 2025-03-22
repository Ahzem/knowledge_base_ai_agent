from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.pandas import PandasTools
from agents.data_summary_agent import DataSummaryAgent
from dotenv import load_dotenv
import json
import pandas as pd
from datetime import datetime
import os

load_dotenv()

class FinancialAnalyticsAgent:
    def __init__(self):
        # Initialize the DataSummaryAgent for direct data access
        self.data_agent = DataSummaryAgent()
        
        # Initialize the Phi Agent but without vector database knowledge
        self.agent = Agent(
            name="Financial Analytics Expert",
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.5,  # Lower temperature for more accuracy
                system=self.get_system_prompt()
            ),
            tools=[
                PandasTools(),
                self._get_financial_metrics,
                self._get_product_metrics,
                self._get_customer_metrics,
                self._get_discount_metrics,
                self._get_period_info
            ],
            description="Expert in financial data analysis and metrics calculation",
            instructions=[
                "Always use the provided metrics from DataSummaryAgent for calculations",
                "Present data in tables when applicable",
                "Include period-over-period comparisons when data is available",
                "Provide specific numbers and percentages",
                "Format currency values appropriately"
            ],
            add_history_to_messages=True,
            num_history_responses=3,
            show_tool_calls=False,
            markdown=True,
        )
    
    def _get_financial_metrics(self, specific_metric=None):
        """Tool to get financial metrics from DataSummaryAgent"""
        metrics = self.data_agent.data_summary.get("financial_metrics", {})
        if specific_metric:
            return {specific_metric: metrics.get(specific_metric)}
        return metrics
    
    def _get_product_metrics(self, specific_category=None):
        """Tool to get product metrics from DataSummaryAgent"""
        metrics = self.data_agent.data_summary.get("product_metrics", {})
        if specific_category == "top_products":
            return {"top_products": metrics.get("top_products", {})}
        elif specific_category == "category_performance":
            return {"category_performance": metrics.get("category_performance", {})}
        return metrics
    
    def _get_customer_metrics(self, specific_metric=None):
        """Tool to get customer metrics from DataSummaryAgent"""
        metrics = self.data_agent.data_summary.get("customer_metrics", {})
        if specific_metric:
            return {specific_metric: metrics.get(specific_metric)}
        return metrics
    
    def _get_discount_metrics(self, specific_metric=None):
        """Tool to get discount metrics from DataSummaryAgent"""
        metrics = self.data_agent.data_summary.get("discount_analysis", {})
        if specific_metric:
            return {specific_metric: metrics.get(specific_metric)}
        return metrics
    
    def _get_period_info(self):
        """Tool to get the current data period"""
        return self.data_agent.data_summary.get("period", {})
    
    def _get_full_dataset_as_dataframe(self):
        """Get the full dataset as DataFrame for complex calculations"""
        if self.data_agent.df is not None:
            return self.data_agent.df
        return pd.DataFrame()  # Return empty DataFrame if no data available

    @staticmethod
    def get_system_prompt():
        return """You are an expert financial analyst specializing in small business metrics. Your answers are based EXCLUSIVELY on pre-calculated metrics provided to you via tools.

For all analyses, prefer using the metrics provided by the DataSummaryAgent tools rather than recalculating from raw data, as these metrics have been carefully pre-calculated for accuracy.

When generating responses:

1. Use The Following Metrics (accessible via tools):
   - Financial metrics (total_revenue, avg_transaction, total_profit, etc.)
   - Product metrics (top products, category performance)
   - Customer metrics (unique customers, repeat rates, top customers)
   - Discount data (discount amounts, rates, refunds)
   - Current period information

2. Always Include:
   - Specific pre-calculated metrics with proper formatting
   - Simple, clear explanations of what the numbers mean
   - Business implications and actionable insights
   - Visual representations described in markdown format when useful

3. Format Your Response With:
   - Clear section headers using markdown
   - Tables for organized data presentation
   - Bullet points for key findings
   - Specific, actionable recommendations

IMPORTANT: Do NOT invent or hallucinate data. If requested information is not available in the pre-calculated metrics, clearly state that the data is not available. Never present calculations based on insufficient data.
"""

    def analyze(self, query, context=None):
        """Perform financial analysis based on the query"""
        try:
            # First ensure data is current
            self.data_agent._load_data()
            
            # Add rich context about available metrics
            enhanced_query = f"""
Analysis Request: {query}

Available Pre-Calculated Metrics:
- Financial: {list(self.data_agent.data_summary.get('financial_metrics', {}).keys())}
- Product: Structure includes top_products and category_performance
- Customer: {list(self.data_agent.data_summary.get('customer_metrics', {}).keys())}
- Discount: {list(self.data_agent.data_summary.get('discount_analysis', {}).keys())}

Current Period: {self.data_agent.data_summary.get('period', {}).get('month')}/{self.data_agent.data_summary.get('period', {}).get('year')}

Additional Context: {context if context else ''}

Please use the available metrics from the tools rather than trying to recalculate values.
"""
            # Run analysis using both pre-calculated metrics and the model
            response = self.agent.run(enhanced_query)
            
            # Format and return the response
            result = response.content if response else "Unable to generate analysis."
            
            # Append a footer with data source information
            period = self.data_agent.data_summary.get('period', {})
            result += f"\n\n---\n*Analysis based on data for {period.get('month')}/{period.get('year')}*"
            
            return result

        except Exception as e:
            print(f"Analysis Error: {str(e)}")
            return f"Error: Unable to complete analysis due to technical issues: {str(e)}"
    
    def get_period_comparison(self, current_period=None, previous_period=None):
        """Get comparison data between two periods"""
        try:
            # Store current summary
            original_summary = self.data_agent.data_summary.copy()
            
            # If no periods specified, compare current with previous month
            if not current_period and not previous_period:
                current_period = (original_summary.get('period', {}).get('year'), 
                                 original_summary.get('period', {}).get('month'))
                
                # Calculate previous month and year
                prev_month = current_period[1] - 1
                prev_year = current_period[0]
                if prev_month < 1:
                    prev_month = 12
                    prev_year -= 1
                
                previous_period = (prev_year, prev_month)
            
            # Get current period data (or use existing if matches)
            if current_period and (current_period[0] != original_summary.get('period', {}).get('year') or
                                  current_period[1] != original_summary.get('period', {}).get('month')):
                self.data_agent.update_data_summary(current_period)
            
            current_data = self.data_agent.data_summary.copy()
            
            # Get previous period data
            self.data_agent.update_data_summary(previous_period)
            previous_data = self.data_agent.data_summary.copy()
            
            # Restore original data
            self.data_agent.data_summary = original_summary
            
            # Create comparison data structure
            comparison = {
                'current_period': current_period,
                'previous_period': previous_period,
                'financial': self._compare_metrics(
                    current_data.get('financial_metrics', {}),
                    previous_data.get('financial_metrics', {})
                ),
                'products': {
                    'current': current_data.get('product_metrics', {}).get('top_products', {}),
                    'previous': previous_data.get('product_metrics', {}).get('top_products', {})
                },
                'customers': self._compare_metrics(
                    current_data.get('customer_metrics', {}),
                    previous_data.get('customer_metrics', {})
                ),
                'discounts': self._compare_metrics(
                    current_data.get('discount_analysis', {}),
                    previous_data.get('discount_analysis', {})
                )
            }
            
            return comparison
            
        except Exception as e:
            print(f"Period comparison error: {str(e)}")
            return None
    
    def _compare_metrics(self, current, previous):
        """Calculate and format comparison between metrics"""
        comparison = {}
        
        for key in current:
            if key in previous and isinstance(current[key], (int, float)) and not key.endswith('_formatted'):
                try:
                    value_current = current[key]
                    value_previous = previous[key]
                    
                    # Calculate change and percentage
                    change = value_current - value_previous
                    pct_change = (change / value_previous * 100) if value_previous != 0 else 0
                    
                    comparison[key] = {
                        'current': value_current,
                        'current_formatted': current.get(f"{key}_formatted", f"{value_current:,.2f}"),
                        'previous': value_previous,
                        'previous_formatted': previous.get(f"{key}_formatted", f"{value_previous:,.2f}"),
                        'change': change,
                        'pct_change': pct_change,
                        'trend': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                    }
                except Exception as e:
                    print(f"Error comparing {key}: {str(e)}")
        
        return comparison
    
    def export_metrics_to_json(self, filepath=None):
        """Export all metrics to a JSON file for reference"""
        if not filepath:
            os.makedirs('exports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"exports/financial_metrics_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.data_agent.data_summary, f, indent=2, default=str)
            return filepath
        except Exception as e:
            print(f"Error exporting metrics: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    analyst = FinancialAnalyticsAgent()
    
    # Test queries
    test_queries = [
        "What are our total revenue and profit margins for this period?",
        "Which products are performing best?",
        "Summarize our customer metrics",
        "How effective are our discounts?",
        "What's our overall business performance for this period?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        print(analyst.analyze(query))