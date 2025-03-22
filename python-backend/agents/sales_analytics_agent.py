from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.csv import CSVUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from agents.web_search_agent import WebSearchAgent
from agents.recommendation_agent import RecommendationAgent
from agents.visualization_agent import VisualizationAgent
import json
from config.api import get_csv_url
from dotenv import load_dotenv
import pandas as pd
import re
import os
from datetime import datetime

load_dotenv()

class SalesFinanceAgent:
    def __init__(self):
        # First, try to load data directly to ensure we have it available
        csv_url = get_csv_url()
        try:
            self.sales_data = pd.read_csv(csv_url)
            print(f"Successfully loaded data with {len(self.sales_data)} records")
            
            # Preprocess data for easier analysis
            self._preprocess_data()
        except Exception as e:
            print(f"Warning: Could not load data directly: {str(e)}")
            self.sales_data = None
        
        # Set up knowledge base with the CSV file - using urls (plural) parameter
        self.knowledge_base = CSVUrlKnowledgeBase(
            urls=[csv_url],  # Use urls parameter with a list
            vector_db=PgVector(
                table_name="sales_finance",
                db_url=os.getenv("DATABASE_URL"),
            ),
        )
        
        self.web_search_agent = Agent(
            name="Web Search Agent",
            description="Expert web search agent that finds relevant and up-to-date information",
            agent=WebSearchAgent()
        )

        # Update model name to correct value
        self.agent = Agent(
            name="Sales Finance Analyst",
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.7,
                system=self.get_system_prompt()
            ),
            description="Expert financial and sales analyst with deep expertise in retail analytics",
            instructions=[
                # Analytical approach
                "Always show calculations for metrics and explain the formulas used",
                "Present numerical comparisons with relevant benchmarks or industry standards",
                "Analyze trends over time with at least 3 data points when available",
                "Highlight anomalies and outliers in the data with possible explanations",
                
                # Data presentation
                "Use markdown tables for structured data presentation", 
                "Include period-over-period comparisons with percentage changes",
                "Use bullet points for key findings and takeaways",
                "Separate raw data from interpretations clearly",
                
                # Insights and recommendations
                "Provide 3-5 specific, actionable insights for each analysis",
                "Prioritize recommendations by potential impact and implementation effort",
                "Balance short-term tactical advice with long-term strategic guidance",
                "Always connect product performance to financial outcomes",
                
                # Response formatting
                "Format responses with clear hierarchical sections using markdown headings",
                "Begin each analysis with a concise executive summary of key findings",
                "End with specific next steps or decision points for the user",
                "Use consistent formatting for all numerical values (e.g., currency, percentages)"
            ],
            team=[self.web_search_agent],
            knowledge=self.knowledge_base,
            add_history_to_messages=True,
            num_history_responses=10,
            search_knowledge=True,
            show_tool_calls=True,
            markdown=True,
        )

    def _preprocess_data(self):
        """Preprocess data for easier analysis"""
        # Convert date column to datetime
        if 'Invoice Date' in self.sales_data.columns:
            self.sales_data['Invoice Date'] = pd.to_datetime(self.sales_data['Invoice Date'], format='%m/%d/%Y', errors='coerce')
        
        # Create month and year columns for easier filtering
        if 'Invoice Date' in self.sales_data.columns:
            self.sales_data['Month'] = self.sales_data['Invoice Date'].dt.month
            self.sales_data['Year'] = self.sales_data['Invoice Date'].dt.year
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['Total Invoice Amount', 'Quantity Sold', 'Sale Price per Unit', 
                         'Purchase Price per Unit', 'Cost of Goods Sold (COGS)', 'Gross Profit per Sale']
        
        for col in numeric_columns:
            if col in self.sales_data.columns:
                self.sales_data[col] = pd.to_numeric(self.sales_data[col], errors='coerce')

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for the agent"""
        return """You are an expert financial and sales analyst with deep expertise in retail analytics. 
        
        ## ANALYTICAL FRAMEWORK
        When analyzing sales data, apply these structured approaches:
    
        1. Sales Performance Metrics:
           - Calculate KPIs: revenue, transaction count, AOV, conversion rate, profit margin
           - Compare period-over-period growth using both absolute and percentage changes
           - Segment analysis by timeframe, category, channel, and customer segment
           - Identify statistical outliers and their business impact
    
        2. Product Performance Analysis:
           - Identify top/bottom performers by volume, revenue, and profit margin
           - Calculate product contribution to overall portfolio profitability
           - Analyze price elasticity and optimal price points
           - Identify cross-selling and product affinity opportunities
    
        3. Customer Behavior Analysis:
           - Segment customers by RFM (Recency, Frequency, Monetary value)
           - Calculate precise CLV using discounted cash flow model
           - Analyze retention, churn, and reactivation rates by cohort
           - Identify high-value customer characteristics for targeted strategies
    
        4. Inventory Optimization:
           - Calculate key metrics: turnover ratio, days of supply, holding costs
           - Identify stock management issues (overstock, stockouts, slow-moving items)
           - Recommend reorder quantities and safety stock levels
           - Analyze carrying costs versus stockout costs
    
        5. Trend and Forecast Analysis:
           - Decompose time series data into trend, seasonality, and cyclical components
           - Apply appropriate forecasting models based on data patterns
           - Assess forecast accuracy with error metrics (MAPE, RMSE)
           - Provide confidence intervals for projections
    
        6. Pricing and Discount Strategy:
           - Measure price sensitivity and discount elasticity
           - Analyze promotion ROI and cannibalization effects
           - Recommend optimal discount strategies by product/customer segment
           - Quantify margin impact of pricing changes
    
        ## RESPONSE STRUCTURE
        Format all analyses with:
        1. Executive Summary: Key findings and recommendations in 3-5 bullet points
        2. Data Context: Time period, data sources, and limitations
        3. Detailed Analysis: Each relevant metric with calculations shown
        4. Business Implications: What the numbers mean for the business
        5. Actionable Recommendations: Prioritized by impact and implementation effort
        6. Next Steps: Specific actions and follow-up analyses
    
        Always use data-driven insights to support recommendations. Acknowledge data limitations when present. Prioritize insights by business impact rather than just statistical significance.
        """

    def get_data_for_period(self, month, year):
        """Extract data for specific month and year from the dataset"""
        if self.sales_data is None:
            return None
        
        try:
            # Filter for the specific month and year using our preprocessed data
            filtered_data = self.sales_data[
                (self.sales_data['Month'] == month) & 
                (self.sales_data['Year'] == year)
            ]
            
            if len(filtered_data) == 0:
                print(f"No data found for {month}/{year}")
                return None
                
            return filtered_data
        except Exception as e:
            print(f"Error filtering data for {month}/{year}: {str(e)}")
            return None

    def analyze_product_data(self, month=None, year=None):
        """Analyze product performance and profitability with optional time filtering"""
        if self.sales_data is None:
            return "No data available for product analysis"
        
        try:
            # Create a filtered dataset if month/year are specified
            if year is not None:
                if month is not None:
                    # Filter by both month and year
                    filtered_data = self.sales_data[
                        (self.sales_data['Month'] == month) & 
                        (self.sales_data['Year'] == year)
                    ]
                    time_period = f"for {month}/{year}"
                else:
                    # Filter by year only
                    filtered_data = self.sales_data[self.sales_data['Year'] == year]
                    time_period = f"for year {year}"
                
                if len(filtered_data) == 0:
                    return f"No data available for {time_period}"
                dataset = filtered_data
            else:
                dataset = self.sales_data
                time_period = "across all time periods"
            
            print(f"Analyzing data {time_period} - found {len(dataset)} records")

            if 'Product Name' in dataset.columns:
                dataset['Product Name'] = dataset['Product Name'].str.strip().str.title()
            
            # Use the appropriate unique identifier for products
            product_key = 'Product ID' if 'Product ID' in dataset.columns else 'Product Name'
            
            # This ensures we're combining all instances of the same product
            product_metrics = dataset.groupby(product_key).agg({
                'Product Name': 'first',  # Keep the name for display
                'Product Category': 'first',  # Keep the category 
                'Quantity Sold': 'sum',  # Sum all quantities of the same product
                'Total Invoice Amount': 'sum',  # Sum all revenue from the same product
                'Cost of Goods Sold (COGS)': 'sum',  # Sum all costs for the same product
                'Gross Profit per Sale': 'sum',  # Sum all profits for the same product
                'Invoice ID': 'nunique'  # Count unique invoices
            }).reset_index()
            
            # Calculate profit margin
            product_metrics['Profit Margin (%)'] = (
                (product_metrics['Gross Profit per Sale'] / product_metrics['Total Invoice Amount']) * 100
            )
            
            # Calculate average sale price
            product_metrics['Average Sale Price'] = (
                product_metrics['Total Invoice Amount'] / product_metrics['Quantity Sold']
            )
            
            # Sort by quantity sold to find top-selling products
            top_products_by_volume = product_metrics.sort_values(by='Quantity Sold', ascending=False).head(10)
            
            # Sort by total revenue to find top revenue-generating products
            top_products_by_revenue = product_metrics.sort_values(by='Total Invoice Amount', ascending=False).head(10)
            
            # Sort by profit margin to find most profitable products
            top_products_by_margin = product_metrics.sort_values(by='Profit Margin (%)', ascending=False).head(10)
            
            # Add debug printing to verify correct aggregation
            print(f"Successfully aggregated data: found {len(product_metrics)} unique products")
            print(f"Top selling product: {top_products_by_volume.iloc[0]['Product Name']} with {top_products_by_volume.iloc[0]['Quantity Sold']} units")
            
            # Format data for presentation
            product_analysis = {
                'top_by_volume': top_products_by_volume.to_dict(orient='records'),
                'top_by_revenue': top_products_by_revenue.to_dict(orient='records'),
                'top_by_margin': top_products_by_margin.to_dict(orient='records'),
                'time_period': time_period
            }
            
            return product_analysis
        
        except Exception as e:
            print(f"Error analyzing product data: {str(e)}")
            return f"Unable to analyze product data: {str(e)}"

    def analyze_customer_data(self, month=None, year=None):
        """Analyze customer behavior and value with optional time filtering"""
        if self.sales_data is None:
            return "No data available for customer analysis"
        
        try:
            # Create a filtered dataset if month/year are specified
            if month is not None and year is not None:
                filtered_data = self.sales_data[
                    (self.sales_data['Month'] == month) & 
                    (self.sales_data['Year'] == year)
                ]
                if len(filtered_data) == 0:
                    return f"No data available for {month}/{year}"
                dataset = filtered_data
                time_period = f"for {month}/{year}"
            else:
                dataset = self.sales_data
                time_period = "across all time periods"
            
            # Group by customer and calculate metrics
            customer_metrics = dataset.groupby(['Customer ID', 'Customer Name']).agg({
                'Total Invoice Amount': 'sum',
                'Invoice ID': 'count',  # Number of transactions
                'Number of Previous Purchases': 'max',  # This assumes the max is the latest count
                'Gross Profit per Sale': 'sum'
            }).reset_index()
            
            # Calculate average order value
            customer_metrics['Average Order Value'] = (
                customer_metrics['Total Invoice Amount'] / customer_metrics['Invoice ID']
            )
            
            # Calculate a rough CLV estimation based on available data
            # Assuming average customer lifespan of 3 years if not available
            customer_metrics['Estimated CLV'] = customer_metrics['Average Order Value'] * (customer_metrics['Number of Previous Purchases'] + 1)
            
            # Calculate retention rate if possible
            # This is a simplified approach
            total_customers = len(customer_metrics)
            returning_customers = len(customer_metrics[customer_metrics['Number of Previous Purchases'] > 0])
            retention_rate = (returning_customers / total_customers) * 100 if total_customers > 0 else 0
            
            # Sort to find top customers by revenue
            top_customers_by_revenue = customer_metrics.sort_values(by='Total Invoice Amount', ascending=False).head(10)
            
            # Sort to find top customers by transaction count
            top_customers_by_transactions = customer_metrics.sort_values(by='Invoice ID', ascending=False).head(10)
            
            # Sort to find top customers by CLV
            top_customers_by_clv = customer_metrics.sort_values(by='Estimated CLV', ascending=False).head(10)
            
            # Format data for presentation
            customer_analysis = {
                'top_by_revenue': top_customers_by_revenue.to_dict(orient='records'),
                'top_by_transactions': top_customers_by_transactions.to_dict(orient='records'),
                'top_by_clv': top_customers_by_clv.to_dict(orient='records'),
                'overall_retention_rate': retention_rate,
                'total_customers': total_customers,
                'returning_customers': returning_customers,
                'time_period': time_period
            }
            
            return customer_analysis
        
        except Exception as e:
            print(f"Error analyzing customer data: {str(e)}")
            return f"Unable to analyze customer data: {str(e)}"

    def analyze_inventory_data(self, month=None, year=None):
        """Analyze inventory metrics with optional time filtering"""
        if self.sales_data is None:
            return "No data available for inventory analysis"
        
        try:
            # Create a filtered dataset if month/year are specified
            if month is not None and year is not None:
                filtered_data = self.sales_data[
                    (self.sales_data['Month'] == month) & 
                    (self.sales_data['Year'] == year)
                ]
                if len(filtered_data) == 0:
                    return f"No data available for {month}/{year}"
                dataset = filtered_data
                time_period = f"for {month}/{year}"
            else:
                dataset = self.sales_data
                time_period = "across all time periods"
            
            # Group by product and calculate inventory metrics
            inventory_metrics = dataset.groupby(['Product Name', 'Product Category', 'Inventory ID']).agg({
                'Quantity Sold': 'sum',
                'Stock Quantity Available': 'first',  # Current stock level
                'Cost of Goods Sold (COGS)': 'sum'
            }).reset_index()
            
            # Calculate stock turnover ratio (annualized)
            inventory_metrics['Stock Turnover Ratio'] = (
                (inventory_metrics['Quantity Sold'] * 12) / inventory_metrics['Stock Quantity Available']
            )
            
            # Calculate weeks of supply
            inventory_metrics['Weeks of Supply'] = (
                (inventory_metrics['Stock Quantity Available'] / inventory_metrics['Quantity Sold']) * 4
            ) if month else (
                (inventory_metrics['Stock Quantity Available'] / (inventory_metrics['Quantity Sold'] / 12)) * 4
            )
            
            # Identify slow-moving inventory
            slow_moving = inventory_metrics[inventory_metrics['Stock Turnover Ratio'] < 1].sort_values(
                by='Stock Turnover Ratio', ascending=True
            ).head(10)
            
            # Identify fast-moving inventory
            fast_moving = inventory_metrics[inventory_metrics['Stock Turnover Ratio'] > 0].sort_values(
                by='Stock Turnover Ratio', ascending=False
            ).head(10)
            
            # Calculate overall metrics
            avg_turnover = inventory_metrics['Stock Turnover Ratio'].mean()
            avg_weeks_supply = inventory_metrics['Weeks of Supply'].mean()
            
            # Format data for presentation
            inventory_analysis = {
                'slow_moving': slow_moving.to_dict(orient='records'),
                'fast_moving': fast_moving.to_dict(orient='records'),
                'avg_turnover': avg_turnover,
                'avg_weeks_supply': avg_weeks_supply,
                'time_period': time_period
            }
            
            return inventory_analysis
        
        except Exception as e:
            print(f"Error analyzing inventory data: {str(e)}")
            return f"Unable to analyze inventory data: {str(e)}"

    def analyze_sales(self, query, data=None):
        """A single, flexible analysis function that adapts to the query intent"""
        try:
            # 1. Extract key entities and intent from the query
            query_analysis = self._understand_query(query)
            
            # 2. Fetch relevant data based on query understanding
            relevant_data = self._get_relevant_data(query_analysis)
            
            # 3. Perform adaptive analysis based on query intent
            analysis_results = self._analyze_data(query_analysis, relevant_data)
            
            # 4. Generate a response appropriate to the query's complexity level
            return self._generate_response(query, query_analysis, analysis_results)
                
        except Exception as e:
            print(f"Analysis Error: {str(e)}")
            return self._generate_fallback_response(query, str(e))

    def _understand_query(self, query):
        """Extract intent, entities, and analytical needs from the query using AI"""
        # Use the LLM to extract key information from the query
        analysis_prompt = f"""
        Analyze this analytics query to identify key components:
        Query: "{query}"
        
        Extract the following information in JSON format:
        {{
            "intent": "primary intent (e.g., product performance, profitability analysis, comparing time periods)",
            "time_period": {{
                "month": month number (1-12) if specified, null if not mentioned,
                "year": year as number if specified, null if not mentioned
            }},
            "metrics": ["list of metrics mentioned"],
            "format": "simple or detailed",
            "filters": ["any other constraints mentioned"]
        }}
        
        For time periods, map month names to their numerical values:
        January=1, February=2, March=3, April=4, May=5, June=6,
        July=7, August=8, September=9, October=10, November=11, December=12
        
        Only include month and year if explicitly mentioned in the query.
        """
        
        # Use agent to interpret the query intent with improved date extraction
        response = self.agent.run(analysis_prompt) 
        
        # Parse structured response
        try:
            query_analysis = json.loads(response.content)
            
            # Store the original query for reference
            query_analysis["query"] = query
            
            # Print what the AI extracted
            if 'time_period' in query_analysis and isinstance(query_analysis['time_period'], dict):
                month = query_analysis['time_period'].get('month')
                year = query_analysis['time_period'].get('year')
                if month or year:
                    print(f"AI extracted time period: Month={month}, Year={year}")
            
            return query_analysis
            
        except Exception as e:
            print(f"Error parsing AI response: {str(e)}")
            # Fallback if JSON parsing fails - use our own direct extraction
            time_period = self._extract_time_periods(query)
            query_analysis = {
                "intent": self._extract_intent(response.content if response else ""),
                "time_period": time_period,
                "metrics": self._extract_metrics(response.content if response else ""),
                "format": "simple" if "simple" in query.lower() or "just" in query.lower() else "detailed",
                "query": query  # Store the original query for reference
            }
            
            # Print what our regex extracted
            if time_period:
                print(f"Regex extracted time period: Month={time_period.get('month')}, Year={time_period.get('year')}")
            
            return query_analysis
    
    def _try_ai_date_extraction(self, query, ai_response):
        """Attempt to extract date information from AI's response"""
        # First try to find structured date mentions in AI's analysis
        month_pattern = r'month(?:\s*number)?\s*[=:]\s*(\d{1,2})'
        year_pattern = r'year(?:\s*number)?\s*[=:]\s*(\d{4})'
        
        month_match = re.search(month_pattern, ai_response, re.IGNORECASE)
        year_match = re.search(year_pattern, ai_response, re.IGNORECASE)
        
        month = int(month_match.group(1)) if month_match else None
        year = int(year_match.group(1)) if year_match else None
        
        # If found both month and year from AI response
        if month and year:
            return {
                "month": month,
                "year": year,
                "target_period": f"{month}/{year}",
                "source": "ai_extraction"
            }
        
        # If only found year from AI response
        elif year:
            return {
                "month": None,
                "year": year,
                "target_period": f"year {year}",
                "source": "ai_extraction"
            }
        
        # Fallback to traditional regex method as last resort
        return self._extract_time_periods(query)
    
    def _extract_intent(self, content):
        """Extract primary intent from unstructured analysis response"""
        # Common intent keywords mapped to standardized intents
        intent_mapping = {
            "product": "product performance",
            "sell": "product performance",
            "top": "product performance", 
            "customer": "customer analysis",
            "retention": "customer analysis",
            "clv": "customer lifetime value",
            "lifetime": "customer lifetime value",
            "inventory": "inventory analysis",
            "stock": "inventory analysis", 
            "turnover": "inventory analysis",
            "seasonal": "seasonal analysis",
            "trend": "trend analysis",
            "forecast": "forecasting",
            "discount": "discount strategy",
            "promotion": "discount strategy", 
            "margin": "profitability analysis",
            "profit": "profitability analysis",
            "revenue": "revenue analysis"
        }
        
        # Check for intent keywords in the content
        content_lower = content.lower()
        for keyword, intent in intent_mapping.items():
            if keyword in content_lower:
                return intent
                
        # Default to general analysis if no specific intent found
        return "general sales analysis"
    
    def _extract_time_periods(self, query):
        """Extract time period information from query using improved regex patterns"""
        query_lower = query.lower()
        
        # Check for full month name + year format with optional prepositions
        month_year_pattern = r'(?:on|in|for|during)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
        match = re.search(month_year_pattern, query_lower)
        if match:
            month_name = match.group(1)
            year = int(match.group(2))
            
            # Map month names to numbers
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 
                'may': 5, 'june': 6, 'july': 7, 'august': 8, 
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            
            month = month_map.get(month_name)
            return {
                "month": month,
                "year": year
            }
        
        # Check for abbreviated month name + year
        abbr_month_pattern = r'(?:on|in|for|during)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{4})'
        match = re.search(abbr_month_pattern, query_lower)
        if match:
            month_abbr = match.group(1)
            year = int(match.group(2))
            
            # Map abbreviated month names to numbers
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 
                'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 
                'oct': 10, 'nov': 11, 'dec': 12
            }
            
            month = month_map.get(month_abbr)
            return {
                "month": month,
                "year": year
            }
        
        # Check for year + month format (like "2025 January" or "2025/January")
        year_month_pattern = r'(\d{4})(?:[-/\s]+)(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)'
        match = re.search(year_month_pattern, query_lower)
        if match:
            year = int(match.group(1))
            month_str = match.group(2)
            
            # Complete month map including both full and abbreviated names
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 
                'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5,
                'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
                'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10, 
                'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }
            
            month = month_map.get(month_str)
            return {
                "month": month,
                "year": year
            }
        
        # Match patterns like MM/YYYY, MM-YYYY, MM.YYYY
        date_patterns = [
            r'(\d{1,2})\s*/\s*(\d{4})',  # 01/2025
            r'(\d{1,2})\s*-\s*(\d{4})',   # 01-2025
            r'(\d{1,2})\s*\.\s*(\d{4})'   # 01.2025
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                month = int(match.group(1))
                year = int(match.group(2))
                
                return {
                    "month": month,
                    "year": year
                }
        
        # Extract year only
        year_pattern = r'\b(20\d{2})\b'
        year_match = re.search(year_pattern, query)
        if year_match:
            year = int(year_match.group(1))
            return {
                "year": year,
                "month": None
            }
        
        # Default to null values if no time period found
        return {
            "year": None,
            "month": None
        }
    
    def _extract_metrics(self, content):
        """Extract metrics of interest from analysis content"""
        # Common metrics to look for
        metric_keywords = [
            "revenue", "sales", "profit", "margin", "volume", 
            "transactions", "orders", "customers", "retention",
            "turnover", "inventory", "discount", "price", "cost",
            "quantity", "units", "growth", "average", "trend"
        ]
        
        # Extract mentioned metrics
        content_lower = content.lower()
        mentioned_metrics = [metric for metric in metric_keywords if metric in content_lower]
        
        # Add a priority order based on frequency or position in the content
        metrics_with_priority = []
        for metric in mentioned_metrics:
            count = content_lower.count(metric)
            position = content_lower.find(metric)
            metrics_with_priority.append({
                "name": metric,
                "priority": count * 1000 - position  # Higher count and earlier position = higher priority
            })
        
        # Sort by priority
        metrics_with_priority.sort(key=lambda x: x["priority"], reverse=True)
        
        # Return the metrics in priority order
        return [m["name"] for m in metrics_with_priority]

    def analyze_discount_effectiveness(self, month=None, year=None):
        """Analyze the effectiveness of discounts with optional time filtering"""
        if self.sales_data is None:
            return "No data available for discount analysis"
        
        try:
            # Create a filtered dataset if month/year are specified
            if month is not None and year is not None:
                filtered_data = self.sales_data[
                    (self.sales_data['Month'] == month) & 
                    (self.sales_data['Year'] == year)
                ]
                if len(filtered_data) == 0:
                    return f"No data available for {month}/{year}"
                dataset = filtered_data
                time_period = f"for {month}/{year}"
            else:
                dataset = self.sales_data
                time_period = "across all time periods"
            
            # Group transactions by discount ranges
            dataset['Discount Range'] = pd.cut(
                dataset['Discount Applied'], 
                bins=[0, 10, 25, 50, 100], 
                labels=['0-10%', '10-25%', '25-50%', '50-100%']
            )
            
            # Calculate metrics by discount range
            discount_metrics = dataset.groupby('Discount Range').agg({
                'Total Invoice Amount': 'sum',
                'Invoice ID': 'count',
                'Gross Profit per Sale': 'sum',
                'Discount Applied': 'mean',
                'Markdown/Discount Loss': 'sum'
            }).reset_index()
            
            # Calculate average order value and profit margin
            discount_metrics['Average Order Value'] = (
                discount_metrics['Total Invoice Amount'] / discount_metrics['Invoice ID']
            )
            
            discount_metrics['Profit Margin %'] = (
                (discount_metrics['Gross Profit per Sale'] / discount_metrics['Total Invoice Amount']) * 100
            )
            
            # Calculate conversion rate (simplified - assuming each invoice is one customer)
            total_invoices = dataset['Invoice ID'].nunique()
            discount_metrics['Sales Percentage'] = (
                (discount_metrics['Invoice ID'] / total_invoices) * 100
            )
            
            # Calculate ROI of discounts
            discount_metrics['Discount ROI'] = (
                (discount_metrics['Gross Profit per Sale'] / discount_metrics['Markdown/Discount Loss']) * 100
            )
            
            # Format data for presentation
            discount_analysis = {
                'discount_metrics': discount_metrics.to_dict(orient='records'),
                'total_discounts': dataset['Discount Applied'].sum(),
                'avg_discount': dataset['Discount Applied'].mean(),
                'total_markdown_loss': dataset['Markdown/Discount Loss'].sum(),
                'total_profit': dataset['Gross Profit per Sale'].sum(),
                'time_period': time_period
            }
            
            return discount_analysis
        
        except Exception as e:
            print(f"Error analyzing discount effectiveness: {str(e)}")
            return f"Unable to analyze discount effectiveness: {str(e)}"

    def calculate_clv(self):
        """Calculate Customer Lifetime Value"""
        if self.sales_data is None:
            return "No data available for CLV analysis"
        
        try:
            # Group by customer
            customer_data = self.sales_data.groupby('Customer ID').agg({
                'Total Invoice Amount': 'sum',
                'Invoice ID': 'count',
                'Number of Previous Purchases': 'max'
            }).reset_index()
            
            # Calculate average order value
            customer_data['AOV'] = customer_data['Total Invoice Amount'] / customer_data['Invoice ID']
            
            # Calculate purchase frequency (total purchases / unique customers)
            purchase_frequency = customer_data['Invoice ID'].sum() / len(customer_data)
            
            # Estimate customer value
            customer_data['Customer Value'] = customer_data['AOV'] * customer_data['Invoice ID']
            
            # Estimate retention rate based on previous purchases
            # This is a simplification - better methods would use cohort analysis
            returning_customers = len(customer_data[customer_data['Number of Previous Purchases'] > 0])
            retention_rate = returning_customers / len(customer_data)
            
            # Calculate average customer lifespan (using retention rate)
            # Using the formula 1/(1-r) where r is retention rate
            avg_lifespan = 1 / (1 - retention_rate) if retention_rate < 1 else 5  # Cap at 5 years if retention rate is 1
            
            # Calculate CLV
            avg_clv = customer_data['AOV'].mean() * purchase_frequency * avg_lifespan
            
            # Get top customers by CLV
            customer_data['Estimated CLV'] = customer_data['AOV'] * (customer_data['Number of Previous Purchases'] + 1) * retention_rate
            top_by_clv = customer_data.sort_values(by='Estimated CLV', ascending=False).head(10)
            
            # Format data for presentation
            clv_analysis = {
                'avg_order_value': customer_data['AOV'].mean(),
                'purchase_frequency': purchase_frequency,
                'retention_rate': retention_rate * 100,  # as percentage
                'avg_lifespan': avg_lifespan,
                'avg_clv': avg_clv,
                'top_by_clv': top_by_clv.to_dict(orient='records')
            }
            
            return clv_analysis
        
        except Exception as e:
            print(f"Error calculating CLV: {str(e)}")
            return f"Unable to calculate CLV: {str(e)}"

    def analyze_seasonal_trends(self):
        """Analyze seasonal sales trends and forecast future quarters"""
        if self.sales_data is None:
            return "No data available for seasonal trend analysis"
        
        try:
            # Create a time series by month
            monthly_sales = self.sales_data.groupby(['Year', 'Month']).agg({
                'Total Invoice Amount': 'sum',
                'Invoice ID': 'count',
                'Quantity Sold': 'sum'
            }).reset_index()
            
            # Create a proper date column
            monthly_sales['Date'] = pd.to_datetime(monthly_sales['Year'].astype(str) + '-' + 
                                                monthly_sales['Month'].astype(str) + '-01')
            
            monthly_sales = monthly_sales.sort_values('Date')
            
            # Calculate quarter
            monthly_sales['Quarter'] = monthly_sales['Date'].dt.quarter
            
            # Group by quarter
            quarterly_sales = monthly_sales.groupby(['Year', 'Quarter']).agg({
                'Total Invoice Amount': 'sum',
                'Invoice ID': 'count',
                'Quantity Sold': 'sum'
            }).reset_index()
            
            # Add quarter label
            quarterly_sales['QuarterLabel'] = quarterly_sales['Year'].astype(str) + 'Q' + quarterly_sales['Quarter'].astype(str)
            
            # Calculate growth rates
            quarterly_sales['Revenue_Growth'] = quarterly_sales['Total Invoice Amount'].pct_change() * 100
            quarterly_sales['Sales_Growth'] = quarterly_sales['Invoice ID'].pct_change() * 100
            
            # Simple forecasting for next quarter using moving average (last 4 quarters)
            # Only forecast if we have enough data
            forecast = {}
            if len(quarterly_sales) >= 4:
                last_quarters = quarterly_sales.tail(4)
                forecast = {
                    'next_quarter_revenue': last_quarters['Total Invoice Amount'].mean(),
                    'next_quarter_sales': last_quarters['Invoice ID'].mean(),
                    'next_quarter_quantity': last_quarters['Quantity Sold'].mean(),
                    'next_quarter': f"{int(last_quarters.iloc[-1]['Year'])}Q{int(last_quarters.iloc[-1]['Quarter']) % 4 + 1}"
                }
            
            # Check seasonality by comparing quarters across years
            has_seasonality = False
            seasonal_patterns = []
            
            if len(quarterly_sales) >= 8:  # Need at least 2 years of data
                # Group by quarter to see patterns
                quarter_performance = quarterly_sales.groupby('Quarter').agg({
                    'Total Invoice Amount': 'mean',
                    'Invoice ID': 'mean'
                }).reset_index()
                
                # Identify the highest performing quarter
                best_quarter = quarter_performance.loc[quarter_performance['Total Invoice Amount'].idxmax()]['Quarter']
                worst_quarter = quarter_performance.loc[quarter_performance['Total Invoice Amount'].idxmin()]['Quarter']
                
                # Check if there's a significant difference
                max_revenue = quarter_performance['Total Invoice Amount'].max()
                min_revenue = quarter_performance['Total Invoice Amount'].min()
                
                if (max_revenue - min_revenue) / min_revenue > 0.2:  # 20% difference indicates seasonality
                    has_seasonality = True
                    seasonal_patterns = [
                        {'quarter': int(quarter), 
                        'revenue': row['Total Invoice Amount'], 
                        'sales': row['Invoice ID']} 
                        for quarter, row in quarter_performance.iterrows()
                    ]
            
            # Format data for presentation
            seasonal_analysis = {
                'quarterly_data': quarterly_sales.to_dict(orient='records'),
                'has_seasonality': has_seasonality,
                'seasonal_patterns': seasonal_patterns,
                'forecast': forecast
            }
            
            return seasonal_analysis
        
        except Exception as e:
            print(f"Error analyzing seasonal trends: {str(e)}")
            return f"Unable to analyze seasonal trends: {str(e)}"

    def process_query(self, query, context=None):
        """Process user query using the agent"""
        try:
            # Check if it's a simple greeting or casual question
            simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
            if query.lower() in simple_greetings:
                return f"Hi! How can I help you today?"
            
            # Use the main agent to process the query
            enhanced_query = f"""
            User Query: {query}
            
            Additional Context: {context if context else 'No additional context provided'}
            
            Current Date: {datetime.now().strftime('%Y-%m-%d')}
            """
            
            response = self.agent.run(enhanced_query)
            return response.content
    
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg

    def _get_relevant_data(self, query_analysis):
        """Retrieve and filter data based on query understanding"""
        try:
            # Check if data is available
            if self.sales_data is None:
                return {
                    "success": False, 
                    "error": "No sales data available",
                    "data": None
                }
            
            # Extract time periods from query analysis
            time_period = query_analysis.get('time_period', {})
            
            # Get month and year from time_period
            year = time_period.get('year')
            month = time_period.get('month')
            
            # Print what we're using to filter
            print(f"Filtering data with: Month={month}, Year={year}")
            
            # Filter data based on time period
            filtered_data = self.sales_data.copy()  # Create a copy to avoid SettingWithCopyWarning
            time_desc = "all time periods"
            
            if year is not None:
                filtered_data = filtered_data[filtered_data['Year'] == year]
                time_desc = f"year {year}"
                
                if month is not None:
                    filtered_data = filtered_data[filtered_data['Month'] == month]
                    time_desc = f"{month}/{year}"
                    print(f"Filtering data for {time_desc}")
            
            # If no data after filtering, return error
            if len(filtered_data) == 0:
                return {
                    "success": False,
                    "error": f"No data available for {time_desc}",
                    "data": None,
                    "time_period": time_desc
                }
            
            # Return the filtered dataset with metadata
            print(f"Found {len(filtered_data)} records for {time_desc}")
            return {
                "success": True,
                "data": filtered_data,
                "record_count": len(filtered_data),
                "time_period": time_desc,
                "year": year,
                "month": month
            }
            
        except Exception as e:
            print(f"Error getting relevant data: {str(e)}")
            return {
                "success": False,
                "error": f"Error retrieving data: {str(e)}",
                "data": None
            }

    def _analyze_data(self, query_analysis, data_context):
        """Perform adaptive analysis based on query intent and available data"""
        try:
            # If data retrieval failed, return the error
            if not data_context.get("success", False):
                return {
                    "success": False,
                    "error": data_context.get("error", "Failed to retrieve relevant data"),
                    "insights": None
                }
            
            # Extract primary intent from query analysis
            intent = query_analysis.get("intent", "").lower()
            data = data_context.get("data")
            year = data_context.get("year")
            month = data_context.get("month")
            time_period = data_context.get("time_period")
            
            # Prepare results container
            analysis_results = {
                "success": True,
                "time_period": time_period,
                "metrics": {},
                "insights": [],
                "format": query_analysis.get("format", "detailed")
            }
            
            # Calculate standard metrics for all queries
            if data is not None:
                analysis_results["metrics"]["total_revenue"] = data["Total Invoice Amount"].sum()
                analysis_results["metrics"]["total_transactions"] = data["Invoice ID"].nunique()
                analysis_results["metrics"]["total_units_sold"] = data["Quantity Sold"].sum()
                
                if "Cost of Goods Sold (COGS)" in data.columns:
                    analysis_results["metrics"]["total_cost"] = data["Cost of Goods Sold (COGS)"].sum()
                    analysis_results["metrics"]["gross_profit"] = analysis_results["metrics"]["total_revenue"] - analysis_results["metrics"]["total_cost"]
                    analysis_results["metrics"]["profit_margin"] = (analysis_results["metrics"]["gross_profit"] / analysis_results["metrics"]["total_revenue"]) * 100
            
            # # Perform specific analysis based on intent
            if "product" in intent or "sell" in intent or "top" in intent:
                # Use existing product analysis as a helper function
                product_data = self.analyze_product_data(month, year)
                if isinstance(product_data, dict):
                    analysis_results["product_analysis"] = {
                        "top_by_volume": product_data.get("top_by_volume", [])[:5],
                        "top_by_revenue": product_data.get("top_by_revenue", [])[:5],
                        "top_by_margin": product_data.get("top_by_margin", [])[:5]
                    }
                    
                    # Add insights based on product data
                    if len(product_data.get("top_by_volume", [])) > 0:
                        top_product = product_data["top_by_volume"][0]
                        analysis_results["insights"].append({
                            "type": "top_product",
                            "metric": "volume",
                            "product": top_product.get("Product Name"),
                            "value": top_product.get("Quantity Sold"),
                            "category": top_product.get("Product Category")
                        })
            
            elif "customer" in intent or "retention" in intent or "clv" in intent:
                # Use existing customer analysis
                if "lifetime" in intent or "clv" in intent:
                    customer_data = self.calculate_clv()
                else:
                    customer_data = self.analyze_customer_data(month, year)
                    
                if isinstance(customer_data, dict):
                    analysis_results["customer_analysis"] = customer_data
                    
                    # Add insights
                    if "overall_retention_rate" in customer_data:
                        analysis_results["insights"].append({
                            "type": "customer_retention",
                            "value": customer_data["overall_retention_rate"],
                            "context": f"Based on {customer_data.get('total_customers', 0)} total customers"
                        })
            
            elif "inventory" in intent or "stock" in intent:
                # Use existing inventory analysis
                inventory_data = self.analyze_inventory_data(month, year)
                if isinstance(inventory_data, dict):
                    analysis_results["inventory_analysis"] = inventory_data
                    
                    # Add insights
                    if "avg_turnover" in inventory_data:
                        analysis_results["insights"].append({
                            "type": "inventory_turnover",
                            "value": inventory_data["avg_turnover"],
                            "context": f"Average across all products"
                        })
            
            elif "seasonal" in intent or "trend" in intent or "forecast" in intent:
                # Use existing seasonal analysis
                seasonal_data = self.analyze_seasonal_trends()
                if isinstance(seasonal_data, dict):
                    analysis_results["seasonal_analysis"] = seasonal_data
                    
            elif "discount" in intent or "promotion" in intent:
                # Use existing discount analysis
                discount_data = self.analyze_discount_effectiveness(month, year)
                if isinstance(discount_data, dict):
                    analysis_results["discount_analysis"] = discount_data
                    
            elif "month" in intent and ("profit" in intent or "profitable" in intent):
                
                # Group by month and calculate profitability metrics
                if year is not None:
                    yearly_data = data[data['Year'] == year]
                    monthly_profit = yearly_data.groupby('Month').agg({
                        'Total Invoice Amount': 'sum',
                        'Gross Profit per Sale': 'sum',
                        'Invoice ID': 'nunique'
                    }).reset_index()
                    
                    monthly_profit['Profit Margin (%)'] = (
                        monthly_profit['Gross Profit per Sale'] / monthly_profit['Total Invoice Amount'] * 100
                    )
                    
                    # Find most profitable month
                    if len(monthly_profit) > 0:
                        most_profitable = monthly_profit.sort_values('Gross Profit per Sale', ascending=False).iloc[0]
                        month_name = {
                            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
                        }[most_profitable['Month']]
                        
                        analysis_results["monthly_profitability"] = {
                            "most_profitable_month": month_name,
                            "month_number": int(most_profitable['Month']),
                            "profit": float(most_profitable['Gross Profit per Sale']),
                            "profit_margin": float(most_profitable['Profit Margin (%)']),
                            "revenue": float(most_profitable['Total Invoice Amount']),
                            "all_months": monthly_profit.to_dict(orient='records')
                        }
                        
                        analysis_results["insights"].append({
                            "type": "most_profitable_month",
                            "month": month_name,
                            "profit": float(most_profitable['Gross Profit per Sale']),
                            "profit_margin": float(most_profitable['Profit Margin (%)']),
                            "context": f"Based on data from {year}"
                        })
            
            # If no specific analysis was done but we have data, provide general metrics
            if len(analysis_results["insights"]) == 0 and data is not None:
                # Add general insights about the time period
                analysis_results["insights"].append({
                    "type": "general_performance",
                    "revenue": analysis_results["metrics"]["total_revenue"],
                    "transactions": analysis_results["metrics"]["total_transactions"],
                    "units_sold": analysis_results["metrics"]["total_units_sold"],
                    "time_period": time_period
                })
                
                # Add top product by default
                if "Product Name" in data.columns:
                    top_product = data.groupby("Product Name")["Quantity Sold"].sum().sort_values(ascending=False)
                    if len(top_product) > 0:
                        analysis_results["insights"].append({
                            "type": "top_product_simple",
                            "product": top_product.index[0],
                            "units_sold": top_product.iloc[0]
                        })
                
            return analysis_results
        
        except Exception as e:
            print(f"Error analyzing data: {str(e)}")
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}",
                "insights": None
            }

    def _generate_response(self, query, query_analysis, analysis_results):
        """Create appropriate response based on analysis results and query expectations"""
        try:
            # Check if analysis was successful
            if not analysis_results.get("success", False):
                # Use fallback response if analysis failed
                return self._generate_fallback_response(query, analysis_results.get("error", "Unknown error"))
            
            # Determine response format based on query
            response_format = query_analysis.get("format", "detailed")
            is_simple = response_format == "simple"
            output_type = query_analysis.get("output_type", "text")
            
            # For simple one-line answers (like "top selling product in 2023")
            if is_simple:
                # Extract the most relevant insight
                insights = analysis_results.get("insights", [])
                if not insights:
                    return "I don't have enough information to answer that specific question."
                
                # Get primary insight
                primary_insight = insights[0]
                insight_type = primary_insight.get("type")
                
                # Format simple response based on insight type
                if insight_type == "top_product" or insight_type == "top_product_simple":
                    return f"The top-selling product in {analysis_results['time_period']} is {primary_insight['product']} with {primary_insight.get('value', primary_insight.get('units_sold', 0)):,} units sold."
                    
                elif insight_type == "most_profitable_month":
                    return f"The most profitable month in {primary_insight['context'].split('from ')[1]} is {primary_insight['month']} with ${primary_insight['profit']:,.2f} in profit and a profit margin of {primary_insight['profit_margin']:.2f}%."
                    
                elif insight_type == "customer_retention":
                    return f"The customer retention rate is {primary_insight['value']:.2f}% based on {primary_insight['context']}."
                    
                elif insight_type == "inventory_turnover":
                    return f"The average inventory turnover ratio is {primary_insight['value']:.2f} times per year."
                    
                elif insight_type == "general_performance":
                    return f"For {primary_insight['time_period']}, we had ${primary_insight['revenue']:,.2f} in revenue from {primary_insight['transactions']:,} transactions and {primary_insight['units_sold']:,} units sold."
                
                # Default simple response
                return f"I've analyzed the data for {analysis_results['time_period']} but I'm not sure how to summarize it briefly. Try asking for more specific metrics or request a detailed analysis."
    
            # For detailed responses, use the LLM to generate a comprehensive analysis
            else:
                # Create context from analysis results
                analysis_context = self._format_context_for_llm(analysis_results)
                
                # Formulate prompt for LLM
                detailed_prompt = f"""
                User Query: {query}
                
                Analysis Context:
                {analysis_context}
                
                Time Period: {analysis_results.get('time_period', 'Not specified')}
                
                Based on this analysis, please provide:
                1. A comprehensive answer to the user's query
                2. Key insights from the data
                3. Any relevant trends or patterns
                4. Business recommendations based on these findings
                
                Format your response as a cohesive analysis that directly addresses the user's question.
                Be specific with numbers and metrics where available.
                """
                
                # Get response from LLM
                response = self.agent.run(detailed_prompt)
                return response.content if response else "I couldn't generate a detailed analysis from the available data."
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(query, str(e))
   
    def _generate_fallback_response(self, query, error):
        """Generate a helpful fallback response when analysis fails"""
        # Clean up error message for user display
        user_friendly_error = self._sanitize_error_message(error)
        
        # Check for common error types and provide helpful responses
        if "No data available" in error or "no data found" in error.lower():
            time_info = self._extract_time_from_error(error)
            return f"I don't have any sales data for {time_info or 'the requested time period'}. Please try a different time range or check if the date format is correct."
        
        elif "invalid literal" in error.lower() or "parsing" in error.lower():
            return "I'm having trouble understanding the date in your question. Could you rephrase it with a clearer date format? For example, 'January 2023' or '1/2023'."
        
        elif "column" in error.lower() and "not in" in error.lower():
            return "I can't find some of the data fields needed to answer this question. This might be because the data structure has changed or the specific metrics aren't available."
        
        elif "future" in query.lower() and ("2024" in query or "2025" in query):
            return "I can only analyze historical data, not make predictions about future periods that haven't occurred yet. I can help with historical analysis or general trends from past data."
        
        # For general errors, provide a more generic but still helpful response
        return f"I'm sorry, I couldn't complete that analysis. {user_friendly_error} Could you try rephrasing your question or asking about a different metric?"

    def _sanitize_error_message(self, error):
        """Clean up error messages to be user-friendly"""
        # Remove technical details that aren't helpful to users
        error_str = str(error)
        
        # Remove stack traces and file paths
        if "Traceback" in error_str:
            error_str = error_str.split("Traceback")[0].strip()
        
        # Remove specific Python exceptions
        exception_patterns = [r'[\w\.]+Error:', r'Exception:', r'ValueError:']
        for pattern in exception_patterns:
            error_str = re.sub(pattern, '', error_str).strip()
        
        # If asking about future dates that don't exist in data
        if "year" in error_str.lower() and any(yr in error_str for yr in ["2024", "2025", "2026"]):
            return "I don't have data for future time periods."
        
        # If error message is too long, truncate it
        if len(error_str) > 100:
            error_str = error_str[:100] + "..."
        
        return error_str

    def _extract_time_from_error(self, error):
        """Try to extract time information from error message"""
        # Look for month/year patterns
        month_year_pattern = r'(\d{1,2})/(\d{4})'
        match = re.search(month_year_pattern, error)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        
        # Look for year only
        year_pattern = r'year (\d{4})'
        match = re.search(year_pattern, error, re.IGNORECASE)
        if match:
            return f"year {match.group(1)}"
        
        # Look for month name + year
        month_name_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
        match = re.search(month_name_pattern, error, re.IGNORECASE)
        if match:
            return f"{match.group(1).title()} {match.group(2)}"
        
        return None
                
    def _format_context_for_llm(self, analysis_results):
        """Format analysis results into a context string for the LLM"""
        context = f"# Analysis for {analysis_results.get('time_period', 'All Time')}\n\n"
        
        # Add general metrics
        metrics = analysis_results.get("metrics", {})
        if metrics:
            context += "## General Metrics\n"
            if "total_revenue" in metrics:
                context += f"- Total Revenue: ${metrics['total_revenue']:,.2f}\n"
            if "total_transactions" in metrics:
                context += f"- Total Transactions: {metrics['total_transactions']:,}\n"
            if "total_units_sold" in metrics:
                context += f"- Total Units Sold: {metrics['total_units_sold']:,}\n"
            if "gross_profit" in metrics:
                context += f"- Gross Profit: ${metrics['gross_profit']:,.2f}\n"
            if "profit_margin" in metrics:
                context += f"- Profit Margin: {metrics['profit_margin']:.2f}%\n"
            context += "\n"
        
        # Add product analysis if available
        if "product_analysis" in analysis_results:
            context += "## Product Analysis\n\n"
            
            # Top by volume
            if "top_by_volume" in analysis_results["product_analysis"]:
                context += "### Top Products by Sales Volume\n"
                for i, product in enumerate(analysis_results["product_analysis"]["top_by_volume"][:5], 1):
                    context += f"{i}. {product.get('Product Name')}: {product.get('Quantity Sold'):,} units\n"
                context += "\n"
            
            # Top by revenue
            if "top_by_revenue" in analysis_results["product_analysis"]:
                context += "### Top Products by Revenue\n"
                for i, product in enumerate(analysis_results["product_analysis"]["top_by_revenue"][:5], 1):
                    context += f"{i}. {product.get('Product Name')}: ${product.get('Total Invoice Amount'):,.2f}\n"
                context += "\n"
            
            # Top by margin
            if "top_by_margin" in analysis_results["product_analysis"]:
                context += "### Top Products by Profit Margin\n"
                for i, product in enumerate(analysis_results["product_analysis"]["top_by_margin"][:5], 1):
                    context += f"{i}. {product.get('Product Name')}: {product.get('Profit Margin (%)'):,.2f}%\n"
                context += "\n"
        
        # Add monthly profitability analysis if available
        if "monthly_profitability" in analysis_results:
            mp = analysis_results["monthly_profitability"]
            context += "## Monthly Profitability Analysis\n\n"
            context += f"Most profitable month: {mp.get('most_profitable_month')}\n"
            context += f"Profit: ${mp.get('profit'):,.2f}\n"
            context += f"Profit Margin: {mp.get('profit_margin'):,.2f}%\n"
            context += f"Revenue: ${mp.get('revenue'):,.2f}\n\n"
            
            context += "### Monthly Comparison\n"
            for month_data in mp.get("all_months", []):
                month_name = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }[month_data.get('Month')]
                context += f"- {month_name}: ${month_data.get('Gross Profit per Sale'):,.2f} profit, {month_data.get('Profit Margin (%)'):,.2f}% margin\n"
            context += "\n"
        
        # Add other analyses as needed (customer, inventory, seasonal, discount)
        # (similar pattern to above)
        
        return context

def chat():
    """Interactive chat function"""
    agent = SalesFinanceAgent()
    print("🤖 AI Assistant: Hello! I'm here to help you. Type 'exit' to end the conversation.")
    
    # Configure logging to hide HTTP requests in chat
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("phi").setLevel(logging.WARNING)
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("🤖 AI Assistant: Goodbye! Have a great day!")
            break
            
        if user_input:
            print("\n🤖 AI Assistant:", end=" ")
            
            # Check if this is likely a sales/analytics query
            analytics_keywords = ['sales', 'product', 'revenue', 'profit', 'customer', 
                                 'inventory', 'sell', 'discount', 'margin', 'trend',
                                 'performance', 'analyze', 'report', 'metric', 'financial',
                                 'top', 'best', 'worst']
            
            is_analytics_query = any(keyword in user_input.lower() for keyword in analytics_keywords)
            
            if is_analytics_query:
                # Use the specialized analyze_sales method for analytics queries
                response = agent.analyze_sales(user_input)
            else:
                # Use the general process_query method for other queries
                response = agent.process_query(user_input)
            
            print(response)
            
def initialize_system():
    """Initialize the system and update data"""
    print("Initializing AI assistant system...")
    
    # Configure logging to hide HTTP requests
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("phi").setLevel(logging.WARNING)
    
    # Update shared knowledge base using api.py
    print("Loading knowledge base...")
    url = get_csv_url()
    
    if not url:
        print("Warning: Using fallback data due to knowledge base update failure")
    else:
        print(f"Knowledge base successfully loaded with URL: {url}")
    
    # Initialize main chat agent
    print("Initializing chat agent...")
    return SalesFinanceAgent()

if __name__ == "__main__":
    agent = initialize_system()
    chat()



# # Usage example
# if __name__ == "__main__":
#     analyst = SalesFinanceAgent()
    
#     # Example queries
#     queries = [
#         # "Show me the sales performance metrics for march 2025",
#         # "Show me the sales performance metrics for 9/2025",
#         "What are our top-selling products and their profitability on 2025 january?",
#         "Calculate customer retention rate and CLV",
#         "Analyze our inventory turnover ratio",
#         "Show seasonal sales trends and forecast next quarter",
#         "Evaluate the effectiveness of our discount strategies"
#     ]
    
#     for query in queries:
#         print(f"\nQuery: {query}")
#         print("-" * 50)
#         print(analyst.analyze_sales(query))