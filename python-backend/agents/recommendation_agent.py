from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.pandas import PandasTools
from agents.data_summary_agent import DataSummaryAgent
from agents.financial_analytics_agent import FinancialAnalyticsAgent
from dotenv import load_dotenv
import json
import pandas as pd
from datetime import datetime
import os

load_dotenv()

class RecommendationAgent:
    def __init__(self):
        # Initialize the data agents for direct data access
        self.data_agent = DataSummaryAgent()
        self.financial_agent = FinancialAnalyticsAgent()
        
        # Initialize the Phi Agent without vector database knowledge
        self.agent = Agent(
            name="Business Recommendation Expert",
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.7,
                system=self.get_system_prompt()
            ),
            tools=[
                PandasTools(),
                self._get_financial_metrics,
                self._get_product_metrics,
                self._get_customer_metrics,
                self._get_discount_metrics,
                self._get_period_info,
                self._get_period_comparison,
                self._get_dataframe
            ],
            description="Expert in generating data-driven business recommendations",
            instructions=[
                "Base recommendations on historical data patterns",
                "Consider seasonal trends and market conditions",
                "Prioritize actionable insights",
                "Include expected impact and ROI estimates",
                "Support recommendations with specific data points"
            ],
            add_history_to_messages=True,
            num_history_responses=5,
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
    
    def _get_period_comparison(self, num_periods=3):
        """Tool to get period comparison data for trend analysis"""
        try:
            # Get current period
            current_period = self.data_agent.data_summary.get("period", {})
            current_year = current_period.get("year")
            current_month = current_period.get("month")
            
            # Store results
            comparisons = {}
            
            # Get data for current and previous periods
            for i in range(num_periods):
                if i == 0:
                    period_name = "current"
                    year, month = current_year, current_month
                else:
                    period_name = f"previous_{i}"
                    # Calculate previous period
                    month = current_month - i
                    year = current_year
                    while month <= 0:
                        month += 12
                        year -= 1
                
                # Update summary for this period
                self.data_agent.update_data_summary((year, month))
                
                # Store metrics
                comparisons[period_name] = {
                    "period": {"year": year, "month": month},
                    "financial": self.data_agent.data_summary.get("financial_metrics", {}),
                    "products": self.data_agent.data_summary.get("product_metrics", {}).get("top_products", {}),
                    "customers": self.data_agent.data_summary.get("customer_metrics", {})
                }
            
            # Restore current period data
            self.data_agent.update_data_summary((current_year, current_month))
            
            return comparisons
            
        except Exception as e:
            print(f"Error getting period comparison: {str(e)}")
            return {"error": str(e)}
    
    def _get_dataframe(self, period_filter=None):
        """Tool to get the raw dataframe for deeper analysis"""
        if self.data_agent.df is None:
            return {"error": "No data available"}
        
        df = self.data_agent.df.copy()
        
        # Apply period filter if specified
        if period_filter and 'Invoice Date' in df.columns:
            try:
                year, month = period_filter
                df = df[
                    (pd.to_datetime(df['Invoice Date']).dt.year == year) & 
                    (pd.to_datetime(df['Invoice Date']).dt.month == month)
                ]
            except:
                pass
        
        # Return DataFrame statistics and information
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "summary": df.describe().to_dict(),
            "sample": df.head(5).to_dict('records')
        }

    @staticmethod
    def get_system_prompt():
        return """You are an expert business recommendation system that uses pre-calculated metrics instead of vector database knowledge. Your role is to analyze business data and generate highly specific, actionable recommendations.

For all recommendations:

1. Analyze Data Patterns:
   - Use the pre-calculated metrics from the DataSummaryAgent tools
   - Examine period-over-period trends to identify patterns
   - Compare product, customer, and financial metrics across time periods
   - Identify potential seasonal or cyclical patterns

2. Generate Recommendations For:
   - Product inventory planning (which products to stock more/less of)
   - Pricing optimization strategies (price points, discounts, promotions)
   - Marketing strategy improvements (targeting, campaigns, timing)
   - Customer engagement initiatives (loyalty programs, retention)
   - Business operations enhancements (efficiency, cost reduction)

3. Each Recommendation Must Include:
   - Clear rationale based on specific data points
   - Expected impact quantified (e.g., "estimated 15% revenue increase")
   - Implementation timeline with specific steps
   - Required resources and investment
   - Risk factors and mitigation strategies
   - Success metrics and KPIs to track

4. Format Guidelines:
   - Prioritize recommendations by potential ROI
   - Include specific numbers from the metrics 
   - Provide clear, actionable steps for implementation
   - Use bullet points for clarity and tables for data comparison
   - Highlight key insights with markdown formatting

IMPORTANT: Do NOT invent or hallucinate data. Only use the metrics provided through tools. If you need specific metrics that aren't available, clearly state this limitation rather than making up values.
"""

    def get_recommendations(self, query, context=None):
        """Generate recommendations based on the query"""
        try:
            # First ensure data is current
            self.data_agent._load_data()
            
            # Add rich context about available metrics
            enhanced_query = f"""
Recommendation Request: {query}

Available Pre-Calculated Metrics:
- Financial: {list(self.data_agent.data_summary.get('financial_metrics', {}).keys())}
- Product: Structure includes top_products and category_performance 
- Customer: {list(self.data_agent.data_summary.get('customer_metrics', {}).keys())}
- Discount: {list(self.data_agent.data_summary.get('discount_analysis', {}).keys())}

Current Period: {self.data_agent.data_summary.get('period', {}).get('month')}/{self.data_agent.data_summary.get('period', {}).get('year')}

You have tools to access period comparison data for trend analysis. Use them to inform your recommendations.

Additional Context: {context if context else 'Use available metrics to make data-driven recommendations'}

Please provide specific, actionable recommendations with implementation details, expected impact, and success metrics.
"""
            # Run analysis using pre-calculated metrics
            response = self.agent.run(enhanced_query)
            
            # Format and return the response
            result = response.content if response else "Unable to generate recommendations."
            
            # Append a footer with data source information
            period = self.data_agent.data_summary.get('period', {})
            result += f"\n\n---\n*Recommendations based on data for {period.get('month')}/{period.get('year')}*"
            
            return result

        except Exception as e:
            print(f"Recommendation Error: {str(e)}")
            return f"Error: Unable to generate recommendations due to technical issues: {str(e)}"
    
    def get_targeted_recommendations(self, recommendation_type, context=None):
        """Generate recommendations for a specific business area"""
        recommendation_types = {
            "inventory": "Analyze our product performance and recommend inventory adjustments. Which products should we stock more of, and which should we reduce? Consider seasonality and trends.",
            "pricing": "Analyze our pricing strategy and recommend optimizations. Consider discounts, markups, and price elasticity based on historical sales data.",
            "marketing": "Recommend marketing strategies based on customer segments and product performance. Which products should we promote to which customers?",
            "customer": "Analyze customer behavior and recommend retention and loyalty strategies. Identify at-risk customers and suggest engagement tactics.",
            "operations": "Recommend operational improvements based on financial metrics and efficiency indicators."
        }
        
        if recommendation_type in recommendation_types:
            query = recommendation_types[recommendation_type]
            # Add context specific to the recommendation type
            enhanced_context = f"{context if context else ''} Focus on {recommendation_type}-specific metrics and actionable insights."
            return self.get_recommendations(query, enhanced_context)
        else:
            return f"Invalid recommendation type. Available types: {', '.join(recommendation_types.keys())}"
    
    def export_recommendations_to_json(self, query, filepath=None):
        """Generate recommendations and export to a JSON file"""
        if not filepath:
            os.makedirs('exports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"exports/recommendations_{timestamp}.json"
        
        try:
            # Get recommendations
            recommendations = self.get_recommendations(query)
            
            # Create a structured object
            data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "period": self.data_agent.data_summary.get("period", {}),
                "recommendations": recommendations,
                "metrics": {
                    "financial": self.data_agent.data_summary.get("financial_metrics", {}),
                    "products": self.data_agent.data_summary.get("product_metrics", {}),
                    "customers": self.data_agent.data_summary.get("customer_metrics", {}),
                    "discounts": self.data_agent.data_summary.get("discount_analysis", {})
                }
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return filepath
        except Exception as e:
            print(f"Error exporting recommendations: {str(e)}")
            return None
    
    def get_predictive_recommendations(self, forecast_periods=3, context=None):
        """Generate forward-looking recommendations with forecasts"""
        try:
            # First ensure data is current
            self.data_agent._load_data()
            
            # Get historical period comparisons for trend analysis
            period_data = self._get_period_comparison(6)  # Get 6 months of data for better trend analysis
            
            # Prepare query with trend context
            forecast_query = f"""
Generate predictive recommendations looking forward {forecast_periods} periods.

Based on the historical data trends:
1. Forecast key metrics for the next {forecast_periods} periods
2. Identify emerging opportunities and risks
3. Recommend proactive strategies to capitalize on predicted trends
4. Suggest contingency plans for potential negative scenarios

When making predictions, consider:
- Growth rates from period to period
- Seasonal patterns in the data
- Product lifecycle indicators
- Customer behavior trends
- Market factors provided in context

Additional Context: {context if context else 'Focus on data-driven forecasts and actionable recommendations'}
"""
            # Get comprehensive recommendations with predictions
            return self.get_recommendations(forecast_query, f"Historical trend data available for analysis. Consider seasonality and growth patterns. {context if context else ''}")
            
        except Exception as e:
            print(f"Predictive Recommendation Error: {str(e)}")
            return f"Error: Unable to generate predictive recommendations: {str(e)}"

# Example usage
if __name__ == "__main__":
    recommender = RecommendationAgent()
    
    # Test queries
    test_queries = [
        "What products should we stock more of next month based on historical trends?",
        "Recommend pricing strategies for our top 5 products",
        "Suggest customer engagement strategies based on purchase patterns",
        "What inventory adjustments should we make for the upcoming season?",
        "Recommend marketing campaigns based on customer segmentation",
        "Suggest operational improvements based on current performance metrics"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        print(recommender.get_recommendations(query))
    
    # Test predictive recommendations
    print("\nPredictive Recommendations")
    print("-" * 50)
    print(recommender.get_predictive_recommendations())