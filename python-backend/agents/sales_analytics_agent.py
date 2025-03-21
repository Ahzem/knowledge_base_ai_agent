from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.csv import CSVUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
import re

load_dotenv()

class SalesFinanceAgent:
    def __init__(self):
        # First, try to load data directly to ensure we have it available
        csv_url = "https://data-analyst-ai-agent.s3.us-east-1.amazonaws.com/FanBudget.csv"
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

        # Update model name to correct value
        self.agent = Agent(
            name="Sales Finance Analyst",
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.7,
                system=self.get_system_prompt()
            ),
            description="Expert in sales and financial analysis",
            instructions=[
                "Always show calculations for metrics",
                "Use tables for data presentation",
                "Include period-over-period comparisons",
                "Provide actionable insights"
            ],
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
        return """You are an expert financial and sales analyst. When analyzing sales data, focus on:

        1. Sales Performance Metrics:
        - Calculate total revenue, sales count, AOV
        - Determine sales growth rates between periods
        - Present trends with specific numbers

        2. Product Analysis:
        - Identify top and bottom performing products
        - Calculate profit margins and total profit
        - Analyze product-level metrics

        3. Customer Analysis:
        - Identify valuable customers
        - Calculate retention rates
        - Determine CLV and purchase frequency

        4. Inventory Analysis:
        - Track stock turnover
        - Identify slow-moving items
        - Calculate inventory efficiency metrics

        5. Trend Analysis:
        - Identify seasonal patterns
        - Forecast demand
        - Analyze price sensitivity

        6. Discount Analysis:
        - Measure discount effectiveness
        - Calculate markdown impact
        - Identify optimal pricing strategies

        Format responses with:
        - Clear section headers
        - Specific metrics and calculations
        - Visual representations when useful
        - Actionable recommendations
        """

    def parse_date_query(self, query):
        """Extract date information from query for better context"""
        # Match patterns like MM/YYYY, MM-YYYY, MM.YYYY, or "Month YYYY"
        date_patterns = [
            r'(\d{1,2})\s*/\s*(\d{4})',  # 4/2024 or 04/2024
            r'(\d{1,2})\s*-\s*(\d{4})',  # 4-2024 or 04-2024
            r'(\d{1,2})\s*\.\s*(\d{4})',  # 4.2024 or 04.2024
            r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})'  # April 2024
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                month, year = match.groups()
                
                # Convert text month to number if needed
                if not month.isdigit():
                    month_map = {
                        'jan': 1, 'january': 1, 
                        'feb': 2, 'february': 2,
                        'mar': 3, 'march': 3,
                        'apr': 4, 'april': 4,
                        'may': 5,
                        'jun': 6, 'june': 6,
                        'jul': 7, 'july': 7,
                        'aug': 8, 'august': 8,
                        'sep': 9, 'september': 9,
                        'oct': 10, 'october': 10,
                        'nov': 11, 'november': 11,
                        'dec': 12, 'december': 12
                    }
                    month = month_map.get(month.lower(), 1)
                
                # Format dates for search
                month_int = int(month)
                year_int = int(year)
                
                previous_month = month_int - 1 if month_int > 1 else 12
                previous_year = year_int if month_int > 1 else year_int - 1
                
                return {
                    'target_period': f"{month_int}/{year_int}",
                    'previous_period': f"{previous_month}/{previous_year}",
                    'month': month_int,
                    'year': year_int
                }
        
        return None

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
            
            # Group by product and calculate metrics
            product_metrics = dataset.groupby(['Product Name', 'Product Category']).agg({
                'Quantity Sold': 'sum',
                'Total Invoice Amount': 'sum',
                'Cost of Goods Sold (COGS)': 'sum',
                'Gross Profit per Sale': 'sum',
                'Invoice ID': 'count'  # Number of invoices/transactions
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

    # Add this improved customer analysis method
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

    # Add this improved inventory analysis method
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
        """Process sales analysis queries"""
        try:
            # First, extract any date information from the query
            date_info = self.parse_date_query(query)
            month = date_info['month'] if date_info else None
            year = date_info['year'] if date_info else None
            
            # Parse the query for different types of analysis
            query_lower = query.lower()
            
            # Product analysis
            if 'product' in query_lower or 'sell' in query_lower or 'profitable' in query_lower:
                product_data = self.analyze_product_data(month, year)
                if isinstance(product_data, dict):
                    # Format product analysis data for the agent
                    top_by_volume = product_data['top_by_volume'][:5]  # Limit to top 5 for clarity
                    top_by_revenue = product_data['top_by_revenue'][:5]
                    top_by_margin = product_data['top_by_margin'][:5]
                    time_period = product_data['time_period']
                    
                    # Format for easier consumption
                    product_context = f"""
                    # Product Analysis {time_period}
    
                    ## Top Products by Sales Volume
                    """
                    for i, product in enumerate(top_by_volume, 1):
                        product_context += f"""
                        {i}. {product['Product Name']} ({product['Product Category']}):
                           - Units Sold: {product['Quantity Sold']:,.0f}
                           - Total Revenue: ${product['Total Invoice Amount']:,.2f}
                           - Profit Margin: {product['Profit Margin (%)']:,.2f}%
                           - Gross Profit: ${product['Gross Profit per Sale']:,.2f}
                        """
                    
                    product_context += """
                    ## Top Products by Revenue
                    """
                    for i, product in enumerate(top_by_revenue, 1):
                        product_context += f"""
                        {i}. {product['Product Name']} ({product['Product Category']}):
                           - Total Revenue: ${product['Total Invoice Amount']:,.2f}
                           - Units Sold: {product['Quantity Sold']:,.0f}
                           - Profit Margin: {product['Profit Margin (%)']:,.2f}%
                           - Gross Profit: ${product['Gross Profit per Sale']:,.2f}
                        """
                    
                    product_context += """
                    ## Top Products by Profit Margin
                    """
                    for i, product in enumerate(top_by_margin, 1):
                        product_context += f"""
                        {i}. {product['Product Name']} ({product['Product Category']}):
                           - Profit Margin: {product['Profit Margin (%)']:,.2f}%
                           - Total Revenue: ${product['Total Invoice Amount']:,.2f}
                           - Units Sold: {product['Quantity Sold']:,.0f}
                           - Gross Profit: ${product['Gross Profit per Sale']:,.2f}
                        """
                    
                    # Send the formatted product analysis to the agent
                    enhanced_query = f"""
                    Analysis Request: {query}
                    
                    {product_context}
                    
                    Time Period: {time_period}
                    
                    Analyze this product data to identify:
                    1. Which products are most profitable and why
                    2. Any product categories that perform particularly well
                    3. Recommendations for product mix optimization
                    4. Any potential concerns or opportunities with specific products
                    
                    Format your response with clear sections, specific insights, and actionable recommendations.
                    """
                    
                    response = self.agent.run(enhanced_query)
                    return response.content if response else "Unable to generate product analysis."
            
            # Check if this is a customer analysis query
            # Update this part in your analyze_sales method to handle all the new analysis types
            # This would replace the existing elif statements after the product analysis check
            elif 'customer' in query_lower or 'retention' in query_lower or 'clv' in query_lower:
                if 'clv' in query_lower or 'lifetime' in query_lower:
                    clv_data = self.calculate_clv()
                    if not isinstance(clv_data, str):
                        # Format CLV data for the agent
                        clv_context = f"""
                        # Customer Lifetime Value Analysis
                        
                        ## Key CLV Metrics
                        - Average Order Value: ${clv_data['avg_order_value']:.2f}
                        - Purchase Frequency: {clv_data['purchase_frequency']:.2f} purchases per customer
                        - Customer Retention Rate: {clv_data['retention_rate']:.2f}%
                        - Average Customer Lifespan: {clv_data['avg_lifespan']:.2f} years
                        - Average Customer Lifetime Value: ${clv_data['avg_clv']:.2f}
                        
                        ## Top Customers by Estimated CLV
                        """
                        
                        for i, customer in enumerate(clv_data['top_by_clv'][:5], 1):
                            clv_context += f"""
                            {i}. Customer ID: {customer['Customer ID']}
                               - Estimated CLV: ${customer['Estimated CLV']:.2f}
                               - Average Order Value: ${customer['AOV']:.2f}
                               - Number of Purchases: {customer['Invoice ID']}
                               - Previous Purchases: {customer['Number of Previous Purchases']}
                            """
                        
                        enhanced_query = f"""
                        Analysis Request: {query}
                        
                        {clv_context}
                        
                        Analyze this CLV data to identify:
                        1. The factors contributing to customer lifetime value
                        2. Strategies to improve customer retention
                        3. Recommendations for increasing customer value
                        4. Opportunities to target high-value customers
                        
                        Format your response with clear sections, specific insights, and actionable recommendations.
                        """
                        
                        response = self.agent.run(enhanced_query)
                        return response.content if response else "Unable to generate CLV analysis."
                else:
                    customer_data = self.analyze_customer_data(month, year)
                    if isinstance(customer_data, dict):
                        # Format customer analysis data for the agent
                        top_by_revenue = customer_data['top_by_revenue'][:5]  # Limit to top 5 for clarity
                        top_by_transactions = customer_data['top_by_transactions'][:5]
                        top_by_clv = customer_data['top_by_clv'][:5]
                        time_period = customer_data['time_period']
                        retention_rate = customer_data['overall_retention_rate']
                        total_customers = customer_data['total_customers']
                        returning_customers = customer_data['returning_customers']
                        
                        # Format for easier consumption
                        customer_context = f"""
                        # Customer Analysis {time_period}
                        
                        ## Customer Retention Metrics
                        - Total Customers: {total_customers}
                        - Returning Customers: {returning_customers}
                        - Overall Retention Rate: {retention_rate:.2f}%
                        
                        ## Top Customers by Revenue
                        """
                        for i, customer in enumerate(top_by_revenue, 1):
                            customer_context += f"""
                            {i}. {customer['Customer Name']} ({customer['Customer ID']}):
                               - Total Revenue: ${customer['Total Invoice Amount']:,.2f}
                               - Number of Transactions: {customer['Invoice ID']}
                               - Average Order Value: ${customer['Average Order Value']:,.2f}
                               - Estimated CLV: ${customer['Estimated CLV']:,.2f}
                            """
                            
                        customer_context += """
                        ## Top Customers by Transactions
                        """
                        for i, customer in enumerate(top_by_transactions, 1):
                            customer_context += f"""
                            {i}. {customer['Customer Name']} ({customer['Customer ID']}):
                               - Number of Transactions: {customer['Invoice ID']}
                               - Total Revenue: ${customer['Total Invoice Amount']:,.2f}
                               - Average Order Value: ${customer['Average Order Value']:,.2f}
                               - Estimated CLV: ${customer['Estimated CLV']:,.2f}
                            """
                            
                        customer_context += """
                        ## Top Customers by Estimated CLV
                        """
                        for i, customer in enumerate(top_by_clv, 1):
                            customer_context += f"""
                            {i}. {customer['Customer Name']} ({customer['Customer ID']}):
                               - Estimated CLV: ${customer['Estimated CLV']:,.2f}
                               - Total Revenue: ${customer['Total Invoice Amount']:,.2f}
                               - Number of Transactions: {customer['Invoice ID']}
                               - Average Order Value: ${customer['Average Order Value']:,.2f}
                            """
                            
                        # Send the formatted customer analysis to the agent

                        enhanced_query = f"""
                        Analysis Request: {query}
                        
                        {customer_context}
                        
                        Time Period: {time_period}
                        
                        Analyze this customer data to identify:
                        1. The most valuable customers and their characteristics
                        2. Trends in customer retention and loyalty
                        3. Strategies to improve customer lifetime value
                        4. Opportunities for personalized marketing
                        
                        Format your response with clear sections, specific insights, and actionable recommendations.
                        """
                        
                        response = self.agent.run(enhanced_query)
                        return response.content if response else "Unable to generate customer analysis."
                    
            # Inventory analysis
            elif 'inventory' in query_lower or 'stock' in query_lower or 'turnover' in query_lower:
                inventory_data = self.analyze_inventory_data(month, year)
                if isinstance(inventory_data, dict):
                    # Format inventory analysis data for the agent
                    slow_moving = inventory_data['slow_moving'][:5]
                    fast_moving = inventory_data['fast_moving'][:5]
                    avg_turnover = inventory_data['avg_turnover']
                    avg_weeks_supply = inventory_data['avg_weeks_supply']
                    time_period = inventory_data['time_period']
                    
                    # Format for easier consumption
                    inventory_context = f"""
                    # Inventory Analysis {time_period}
                    
                    ## Inventory Turnover Metrics
                    - Average Stock Turnover Ratio: {avg_turnover:.2f} per year
                    - Average Weeks of Supply: {avg_weeks_supply:.2f} weeks
                    
                    ## Slow-Moving Inventory
                    """
                    for i, product in enumerate(slow_moving, 1):
                        inventory_context += f"""
                        {i}. {product['Product Name']} ({product['Product Category']}):
                           - Stock Turnover Ratio: {product['Stock Turnover Ratio']:.2f}
                           - Stock Available: {product['Stock Quantity Available']}
                           - COGS: ${product['Cost of Goods Sold (COGS)']:,.2f}
                        """
                        
                    inventory_context += """
                    ## Fast-Moving Inventory
                    """
                    for i, product in enumerate(fast_moving, 1):
                        inventory_context += f"""
                        {i}. {product['Product Name']} ({product['Product Category']}):
                           - Stock Turnover Ratio: {product['Stock Turnover Ratio']:.2f}
                           - Stock Available: {product['Stock Quantity Available']}
                           - COGS: ${product['Cost of Goods Sold (COGS)']:,.2f}
                        """
                        
                    # Send the formatted inventory analysis to the agent
                    enhanced_query = f"""
                    Analysis Request: {query}
                    
                    {inventory_context}
                    
                    Time Period: {time_period}
                    
                    Analyze this inventory data to identify:
                    1. Slow-moving and fast-moving products
                    2. Inventory turnover efficiency
                    3. Recommendations for stock optimization
                    4. Insights on inventory management strategies
                    
                    Format your response with clear sections, specific insights, and actionable recommendations.
                    """
                    
                    response = self.agent.run(enhanced_query)
                    return response.content if response else "Unable to generate inventory analysis."
                
            # For the seasonal analysis section:
            elif 'seasonal' in query_lower or 'trend' in query_lower or 'forecast' in query_lower:
                seasonal_data = self.analyze_seasonal_trends()
                if isinstance(seasonal_data, dict):
                    # Format seasonal data for the agent
                    quarterly_data = seasonal_data['quarterly_data'][:8]  # Show up to 8 quarters
                    has_seasonality = seasonal_data['has_seasonality']
                    seasonal_patterns = seasonal_data['seasonal_patterns']
                    forecast = seasonal_data.get('forecast', {})
                    
                    # Format for easier consumption
                    seasonal_context = """
                    # Seasonal Sales Analysis and Forecast
                    
                    ## Quarterly Performance
                    """
                    
                    for quarter in quarterly_data:
                        quarter_label = quarter['QuarterLabel']
                        revenue = quarter['Total Invoice Amount']
                        sales_count = quarter['Invoice ID']
                        revenue_growth = quarter.get('Revenue_Growth', 'N/A')
                        if isinstance(revenue_growth, float):
                            revenue_growth = f"{revenue_growth:.2f}%"
                        
                        seasonal_context += f"""
                        ### {quarter_label}
                        - Total Revenue: ${revenue:,.2f}
                        - Number of Sales: {sales_count}
                        - Revenue Growth: {revenue_growth}
                        """
                    
                    # Add seasonality insights
                    seasonal_context += """
                    ## Seasonality Analysis
                    """
                    
                    if has_seasonality:
                        seasonal_context += "Significant seasonal patterns detected in the sales data:\n"
                        for pattern in seasonal_patterns:
                            quarter_num = pattern['quarter']
                            quarter_name = f"Q{quarter_num}"
                            seasonal_context += f"- {quarter_name}: ${pattern['revenue']:,.2f} average revenue, {pattern['sales']:.0f} average transactions\n"
                    else:
                        seasonal_context += "No significant seasonal patterns detected in the available data.\n"
                    
                    # Add forecast if available
                    if forecast:
                        seasonal_context += f"""
                        ## Sales Forecast for {forecast['next_quarter']}
                        - Projected Revenue: ${forecast['next_quarter_revenue']:,.2f}
                        - Projected Number of Sales: {forecast['next_quarter_sales']:.0f}
                        - Projected Units Sold: {forecast['next_quarter_quantity']:,.0f}
                        
                        This forecast is based on a moving average of the last 4 quarters.
                        """
                    
                    # Send the formatted seasonal analysis to the agent
                    enhanced_query = f"""
                    Analysis Request: {query}
                    
                    {seasonal_context}
                    
                    Based on this seasonal sales data, please provide:
                    1. An analysis of the sales trends and patterns observed
                    2. Insights into the potential reasons for the observed seasonality
                    3. Assessment of the reliability of the forecast
                    4. Strategic recommendations for managing seasonal fluctuations
                    5. Suggestions for inventory planning based on seasonal patterns
                    
                    Format your response with clear sections, specific insights, and actionable recommendations.
                    """
                    
                    response = self.agent.run(enhanced_query)
                    return response.content if response else "Unable to generate seasonal analysis."
            
            # For the discount analysis section:
            elif 'discount' in query_lower or 'markdown' in query_lower or 'promotion' in query_lower:
                discount_data = self.analyze_discount_effectiveness(month, year)
                if isinstance(discount_data, dict):
                    # Format discount data for the agent
                    discount_metrics = discount_data['discount_metrics']
                    time_period = discount_data['time_period']
                    total_discounts = discount_data['total_discounts']
                    avg_discount = discount_data['avg_discount']
                    total_markdown_loss = discount_data['total_markdown_loss']
                    total_profit = discount_data['total_profit']
                    
                    # Format for easier consumption
                    discount_context = f"""
                    # Discount Strategy Analysis {time_period}
                    
                    ## Overall Discount Metrics
                    - Total Discount Amount: ${total_discounts:,.2f}
                    - Average Discount Applied: {avg_discount:.2f}%
                    - Total Markdown Loss: ${total_markdown_loss:,.2f}
                    - Total Gross Profit: ${total_profit:,.2f}
                    - Overall Discount ROI: {(total_profit / total_markdown_loss * 100) if total_markdown_loss > 0 else 'N/A'}%
                    
                    ## Discount Performance by Range
                    """
                    
                    for range_metrics in discount_metrics:
                        discount_range = range_metrics['Discount Range']
                        revenue = range_metrics['Total Invoice Amount']
                        transactions = range_metrics['Invoice ID']
                        avg_order_value = range_metrics['Average Order Value']
                        profit_margin = range_metrics['Profit Margin %']
                        sales_percentage = range_metrics['Sales Percentage']
                        discount_roi = range_metrics['Discount ROI']
                        
                        discount_context += f"""
                        ### {discount_range} Discount Range
                        - Total Revenue: ${revenue:,.2f}
                        - Number of Transactions: {transactions}
                        - Average Order Value: ${avg_order_value:,.2f}
                        - Profit Margin: {profit_margin:.2f}%
                        - Percentage of Total Sales: {sales_percentage:.2f}%
                        - Discount ROI: {discount_roi:.2f}%
                        """
                    
                    # Send the formatted discount analysis to the agent
                    enhanced_query = f"""
                    Analysis Request: {query}
                    
                    {discount_context}
                    
                    Analyze this discount data to identify:
                    1. Which discount ranges provide the best return on investment
                    2. The impact of discounts on average order value and profit margins
                    3. The effectiveness of current discount strategies
                    4. Recommendations for optimizing discount approaches
                    5. Potential risks or concerns in current discount practices
                    
                    Format your response with clear sections, specific insights, and actionable recommendations.
                    """
                    
                    response = self.agent.run(enhanced_query)
                    return response.content if response else "Unable to generate discount analysis."
            
            # Otherwise, try to extract date information for period analysis
            else:
                date_info = self.parse_date_query(query)
                date_context = ""
                filtered_data = None
                
                if date_info:
                    print(f"Found date in query: {date_info['target_period']}")
                    # Try to get actual data for the periods
                    target_data = self.get_data_for_period(date_info['month'], date_info['year'])
                    previous_month = int(date_info['previous_period'].split('/')[0])
                    previous_year = int(date_info['previous_period'].split('/')[1])
                    previous_data = self.get_data_for_period(previous_month, previous_year)
                    
                    date_context = f"""
                    Target analysis period: {date_info['target_period']}
                    Previous period for comparison: {date_info['previous_period']}
                    
                    When searching for data, look for entries from month {date_info['month']} in year {date_info['year']}.
                    Check 'Invoice Date' column entries in the format DD/MM/YYYY.
                    """
                    
                    # If we have direct access to data, provide metrics immediately
                    if target_data is not None:
                        print(f"Found {len(target_data)} records for {date_info['target_period']}")
                        target_metrics = {
                            'total_revenue': target_data['Total Invoice Amount'].sum(),
                            'num_sales': len(target_data),
                            'avg_order_value': target_data['Total Invoice Amount'].sum() / len(target_data) if len(target_data) > 0 else 0,
                            'total_cogs': target_data['Cost of Goods Sold (COGS)'].sum() if 'Cost of Goods Sold (COGS)' in target_data.columns else None,
                            'month': date_info['month'],
                            'year': date_info['year']
                        }
                        
                        prev_metrics = None
                        if previous_data is not None and len(previous_data) > 0:
                            print(f"Found {len(previous_data)} records for {previous_month}/{previous_year}")
                            prev_metrics = {
                                'total_revenue': previous_data['Total Invoice Amount'].sum(),
                                'num_sales': len(previous_data),
                                'avg_order_value': previous_data['Total Invoice Amount'].sum() / len(previous_data) if len(previous_data) > 0 else 0,
                                'total_cogs': previous_data['Cost of Goods Sold (COGS)'].sum() if 'Cost of Goods Sold (COGS)' in previous_data.columns else None,
                                'month': previous_month,
                                'year': previous_year
                            }
                        
                        # Add data metrics to query context
                        if prev_metrics:
                            # Calculate growth rates if previous data exists
                            revenue_growth = ((target_metrics['total_revenue'] - prev_metrics['total_revenue']) / prev_metrics['total_revenue'] * 100) if prev_metrics['total_revenue'] > 0 else 0
                            sales_growth = ((target_metrics['num_sales'] - prev_metrics['num_sales']) / prev_metrics['num_sales'] * 100) if prev_metrics['num_sales'] > 0 else 0
                            
                            date_context += f"""
                            Target Period Metrics:
                            - Total Revenue: ${target_metrics['total_revenue']:.2f}
                            - Number of Sales: {target_metrics['num_sales']}
                            - Average Order Value: ${target_metrics['avg_order_value']:.2f}
                            
                            Previous Period Metrics:
                            - Total Revenue: ${prev_metrics['total_revenue']:.2f}
                            - Number of Sales: {prev_metrics['num_sales']}
                            - Average Order Value: ${prev_metrics['avg_order_value']:.2f}
                            
                            Growth Rates:
                            - Revenue Growth: {revenue_growth:.2f}%
                            - Sales Count Growth: {sales_growth:.2f}%
                            """
                        else:
                            date_context += f"""
                            Target Period Metrics:
                            - Total Revenue: ${target_metrics['total_revenue']:.2f}
                            - Number of Sales: {target_metrics['num_sales']}
                            - Average Order Value: ${target_metrics['avg_order_value']:.2f}
                            
                            No data available for previous period comparison.
                            """
                    else:
                        date_context += "\nNo direct data found for the specified period. Using knowledge base search instead."
            
            # Enhance query with data context if provided
            enhanced_query = f"""
            Analysis Request: {query}
            
            {date_context if 'date_context' in locals() else ''}
            
            If specific calculations are needed, use these formulas:
            - AOV = Total Revenue / Number of Sales
            - Growth Rate = (Current - Previous) / Previous * 100
            - CLV = AOV × Purchase Frequency × Retention Rate
            - Stock Turnover = COGS / Average Inventory
            
            Data Context: {data if data else 'Use historical data from knowledge base'}
            
            If asked about a specific time period, be sure to filter data from that period only.
            For date fields, check the 'Invoice Date' column.
            
            Provide detailed analysis with specific metrics and recommendations.
            """
            
            response = self.agent.run(enhanced_query)
            return response.content if response else "Unable to generate analysis."

        except Exception as e:
            print(f"Analysis Error: {str(e)}")
            return f"Error: Unable to complete analysis due to technical issues: {str(e)}"

    # Add this new method for discount analysis
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

    # Add this method for CLV analysis
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

    # Add this method for seasonal trend analysis
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

# Usage example
if __name__ == "__main__":
    analyst = SalesFinanceAgent()
    
    # Example queries
    queries = [
        # "Show me the sales performance metrics for march 2025",
        # "Show me the sales performance metrics for 9/2025",
        "What are our top-selling products and their profitability on 2025 january?",
        "Calculate customer retention rate and CLV",
        "Analyze our inventory turnover ratio",
        "Show seasonal sales trends and forecast next quarter",
        "Evaluate the effectiveness of our discount strategies"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        print(analyst.analyze_sales(query))