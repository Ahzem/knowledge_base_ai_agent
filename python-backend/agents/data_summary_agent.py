from datetime import datetime
import pandas as pd
import numpy as np
import os
import requests
from io import StringIO
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.pandas import PandasTools
from config.data_manager import data_manager
from dotenv import load_dotenv

load_dotenv()

class DataSummaryAgent:
    def __init__(self):
        self.data_summary = {}
        self.df = None
        
        # Load data on initialization
        self._load_data()
        
        # Initialize the AI agent for flexible question answering
        self.agent = Agent(
            name="Data Summary Expert",
            model=OpenAIChat(
                model="gpt-4o",
                temperature=0.3,
                system=self.get_system_prompt()
            ),
            tools=[
                PandasTools(),
                self._get_dataframe,
                self._get_financial_metrics,
                self._get_product_metrics,
                self._get_customer_metrics,
                self._get_discount_metrics,
                self._get_period_info
            ],
            description="Expert in analyzing and summarizing business data",
            instructions=[
                "Use pandas to analyze the provided dataframe when needed",
                "Always provide specific numbers and metrics",
                "Filter data by date ranges when specified",
                "Present data in tables when appropriate",
                "Format numbers properly (currency, percentages, etc.)"
            ],
            add_history_to_messages=True,
            show_tool_calls=False,
            markdown=True,
        )
    
    def _load_data(self):
        """Load and process data directly from CSV source"""
        try:
            # Try to fetch data from URLs first
            urls = data_manager.get_urls()
            
            if urls and len(urls) > 0:
                # Use the first URL from the list
                print(f"Fetching data from URL: {urls[0]}")
                try:
                    response = requests.get(urls[0])
                    if response.status_code == 200:
                        self.df = pd.read_csv(StringIO(response.text))
                        print(f"Successfully loaded data with {len(self.df)} rows from URL")
                    else:
                        print(f"Failed to fetch data from URL: {response.status_code}")
                        self._load_local_data()
                except Exception as e:
                    print(f"Error fetching data from URL: {str(e)}")
                    self._load_local_data()
            else:
                self._load_local_data()
            
            # Process data and generate summary
            self._generate_data_summary()
            
        except Exception as e:
            print(f"Error loading data in DataSummaryAgent: {str(e)}")
            self._initialize_empty_summary()
    
    def _load_local_data(self):
        """Load data from local CSV file"""
        local_data_path = 'data/FanBudget.csv'
        if os.path.exists(local_data_path):
            self.df = pd.read_csv(local_data_path)
            print(f"Loaded data from {local_data_path}")
        else:
            print("No local data file found. Using empty dataframe.")
            self.df = pd.DataFrame()
    
    def _generate_data_summary(self):
        """Generate complete data summary metrics"""
        if self.df is None or self.df.empty:
            self._initialize_empty_summary()
            return
        
        # Ensure data types
        if 'Invoice Date' in self.df.columns:
            self.df['Invoice Date'] = pd.to_datetime(self.df['Invoice Date'], errors='coerce')
        
        # Get current date
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        # Filter for current month
        df_filtered = self.df.copy()
        if 'Invoice Date' in df_filtered.columns:
            df_filtered = df_filtered[
                (df_filtered['Invoice Date'].dt.year == current_year) & 
                (df_filtered['Invoice Date'].dt.month == current_month)
            ]
        
        # If no data for current month, use all available data
        if len(df_filtered) == 0:
            print(f"No data for {current_month}/{current_year}. Using all available data.")
            df_filtered = self.df
        
        # Calculate financial metrics
        self._calculate_financial_metrics(df_filtered)
        
        # Calculate product metrics
        self._calculate_product_metrics(df_filtered)
        
        # Calculate customer metrics
        self._calculate_customer_metrics(df_filtered)
        
        # Calculate discount analysis
        self._calculate_discount_metrics(df_filtered)
        
        # Set period information
        self.data_summary["period"] = {
            "month": current_month,
            "year": current_year
        }
        
        print(f"Data summary updated for {current_month}/{current_year}")
    
    def _calculate_financial_metrics(self, df):
        """Calculate financial metrics from dataframe"""
        # Initialize with zeros
        total_revenue = 0
        num_transactions = len(df)
        avg_transaction = 0
        daily_avg_revenue = 0
        total_profit = 0
        total_cogs = 0
        return_rate = 0
        
        # Calculate if data exists
        if 'Total Invoice Amount' in df.columns:
            total_revenue = df['Total Invoice Amount'].sum()
            avg_transaction = total_revenue / num_transactions if num_transactions > 0 else 0
        
        if 'Gross Profit per Sale' in df.columns:
            total_profit = df['Gross Profit per Sale'].sum()
        
        if 'Cost of Goods Sold (COGS)' in df.columns:
            total_cogs = df['Cost of Goods Sold (COGS)'].sum()
        
        if 'Refund Amount' in df.columns:
            return_rate = (df['Refund Amount'] > 0).mean() * 100
        
        # Calculate daily average revenue
        if 'Invoice Date' in df.columns:
            unique_days = df['Invoice Date'].dt.date.nunique()
            daily_avg_revenue = total_revenue / unique_days if unique_days > 0 else 0
        
        # Store financial metrics
        self.data_summary["financial_metrics"] = {
            "total_revenue": total_revenue,
            "total_revenue_formatted": self.format_currency(total_revenue),
            "total_transactions": num_transactions,
            "avg_transaction": avg_transaction,
            "avg_transaction_formatted": self.format_currency(avg_transaction),
            "daily_avg_revenue": daily_avg_revenue,
            "daily_avg_revenue_formatted": self.format_currency(daily_avg_revenue),
            "total_profit": total_profit,
            "total_profit_formatted": self.format_currency(total_profit),
            "total_cogs": total_cogs,
            "total_cogs_formatted": self.format_currency(total_cogs),
            "return_rate": return_rate,
            "return_rate_formatted": self.format_percentage(return_rate)
        }
    
    def _calculate_product_metrics(self, df):
        """Calculate product-related metrics"""
        top_products = {}
        category_performance = {}
        
        # Top products by revenue
        if all(col in df.columns for col in ['Product Name', 'Total Invoice Amount']):
            product_summary = df.groupby('Product Name').agg({
                'Total Invoice Amount': 'sum',
                'Quantity Sold': 'sum' if 'Quantity Sold' in df.columns else lambda x: 0,
                'Gross Profit per Sale': 'sum' if 'Gross Profit per Sale' in df.columns else lambda x: 0
            }).sort_values('Total Invoice Amount', ascending=False)
            
            # Convert to dictionary format for top 5 products
            top_products = product_summary.head(5).to_dict('index')
            
            # Format currency values
            for product, metrics in top_products.items():
                if 'Total Invoice Amount' in metrics:
                    metrics['Total Invoice Amount_formatted'] = self.format_currency(metrics['Total Invoice Amount'])
                if 'Gross Profit per Sale' in metrics:
                    metrics['Gross Profit per Sale_formatted'] = self.format_currency(metrics['Gross Profit per Sale'])
        
        # Category performance
        if all(col in df.columns for col in ['Product Category', 'Total Invoice Amount']):
            category_performance_data = df.groupby('Product Category')['Total Invoice Amount'].sum()
            category_performance = {
                category: {
                    'revenue': amount,
                    'revenue_formatted': self.format_currency(amount)
                } for category, amount in category_performance_data.items()
            }
        
        self.data_summary["product_metrics"] = {
            "top_products": top_products,
            "category_performance": category_performance
        }
    
    def _calculate_customer_metrics(self, df):
        """Calculate customer-related metrics"""
        unique_customers = 0
        avg_customer_purchases = 0
        top_customers = {}
        repeat_rate = 0
        
        if 'Customer ID' in df.columns:
            # Count unique customers
            unique_customers = df['Customer ID'].nunique()
            
            # Calculate average purchases per customer
            customer_purchases = df.groupby('Customer ID').size()
            avg_customer_purchases = customer_purchases.mean() if len(customer_purchases) > 0 else 0
            
            # Calculate repeat rate
            customer_counts = df['Customer ID'].value_counts()
            repeat_customers = len(customer_counts[customer_counts > 1])
            repeat_rate = (repeat_customers / unique_customers) * 100 if unique_customers > 0 else 0
        
        # Top customers by spend
        if all(col in df.columns for col in ['Customer Name', 'Total Invoice Amount']):
            top_customer_data = df.groupby('Customer Name')['Total Invoice Amount'].sum().sort_values(ascending=False)
            top_customers = {
                customer: {
                    'spend': amount,
                    'spend_formatted': self.format_currency(amount)
                } for customer, amount in top_customer_data.head(3).items()
            }
        
        self.data_summary["customer_metrics"] = {
            "unique_customers": unique_customers,
            "avg_customer_purchases": avg_customer_purchases,
            "top_customers": top_customers,
            "repeat_rate": repeat_rate,
            "repeat_rate_formatted": self.format_percentage(repeat_rate)
        }
    
    def _calculate_discount_metrics(self, df):
        """Calculate discount and refund metrics"""
        total_discounts = 0
        total_refunds = 0
        avg_discount_rate = 0
        
        if 'Markdown/Discount Loss' in df.columns:
            total_discounts = df['Markdown/Discount Loss'].sum()
        
        if 'Refund Amount' in df.columns:
            total_refunds = df['Refund Amount'].sum()
        
        if 'Discount Applied' in df.columns:
            avg_discount_rate = df['Discount Applied'].mean() * 100
        
        self.data_summary["discount_analysis"] = {
            "total_discounts": total_discounts,
            "total_discounts_formatted": self.format_currency(total_discounts),
            "total_refunds": total_refunds,
            "total_refunds_formatted": self.format_currency(total_refunds),
            "avg_discount_rate": avg_discount_rate,
            "avg_discount_rate_formatted": self.format_percentage(avg_discount_rate)
        }
    
    def _initialize_empty_summary(self):
        """Initialize with empty data to prevent errors"""
        self.data_summary = {
            "period": {
                "month": datetime.now().month,
                "year": datetime.now().year
            },
            "financial_metrics": {
                "total_revenue": 0,
                "total_revenue_formatted": "$0.00",
                "total_transactions": 0,
                "avg_transaction": 0,
                "avg_transaction_formatted": "$0.00",
                "total_profit": 0,
                "total_profit_formatted": "$0.00",
                "total_cogs": 0,
                "total_cogs_formatted": "$0.00",
                "daily_avg_revenue": 0,
                "daily_avg_revenue_formatted": "$0.00",
                "return_rate": 0,
                "return_rate_formatted": "0.0%"
            },
            "product_metrics": {
                "top_products": {},
                "category_performance": {}
            },
            "customer_metrics": {
                "unique_customers": 0,
                "avg_customer_purchases": 0,
                "top_customers": {},
                "repeat_rate": 0,
                "repeat_rate_formatted": "0.0%"
            },
            "discount_analysis": {
                "total_discounts": 0,
                "total_discounts_formatted": "$0.00",
                "total_refunds": 0,
                "total_refunds_formatted": "$0.00",
                "avg_discount_rate": 0,
                "avg_discount_rate_formatted": "0.0%"
            }
        }

    def _get_revenue_summary(self):
        """Get revenue-specific summary"""
        metrics = self.data_summary['financial_metrics']
        return (
            f"Revenue Summary:\n"
            f"Total Revenue: {metrics['total_revenue_formatted']}\n"
            f"Daily Average Revenue: {metrics['daily_avg_revenue_formatted']}\n"
            f"Number of Transactions: {metrics['total_transactions']:,}\n"
            f"Average Transaction Value: {metrics['avg_transaction_formatted']}"
        )
    
    def _get_product_summary(self):
        """Get product-specific summary"""
        product_metrics = self.data_summary['product_metrics']
        result = "Product Performance Summary:\n\n"
        
        # Top products
        result += "Top Products by Revenue:\n"
        for product, data in product_metrics['top_products'].items():
            result += f"- {product}: {data.get('Total Invoice Amount_formatted', '$0.00')}"
            if 'Quantity Sold' in data:
                result += f" (Qty: {int(data['Quantity Sold']):,})"
            result += "\n"
        
        # Category performance
        result += "\nCategory Performance:\n"
        for category, data in product_metrics['category_performance'].items():
            result += f"- {category}: {data['revenue_formatted']}\n"
        
        return result
    
    def _get_customer_summary(self):
        """Get customer-specific summary"""
        customer_metrics = self.data_summary['customer_metrics']
        return (
            f"Customer Summary:\n"
            f"Unique Customers: {customer_metrics['unique_customers']:,}\n"
            f"Average Purchases per Customer: {customer_metrics['avg_customer_purchases']:.2f}\n"
            f"Customer Repeat Rate: {customer_metrics['repeat_rate_formatted']}\n\n"
            f"Top Customers by Spend:\n" + 
            "\n".join([f"- {customer}: {data['spend_formatted']}" 
                      for customer, data in customer_metrics['top_customers'].items()])
        )
    
    def _get_discount_summary(self):
        """Get discount and refunds summary"""
        discount_metrics = self.data_summary['discount_analysis']
        return (
            f"Discount and Refund Summary:\n"
            f"Total Discounts: {discount_metrics['total_discounts_formatted']}\n"
            f"Average Discount Rate: {discount_metrics['avg_discount_rate_formatted']}\n"
            f"Total Refunds: {discount_metrics['total_refunds_formatted']}"
        )
    
    def _get_profit_summary(self):
        """Get profit-related summary"""
        metrics = self.data_summary['financial_metrics']
        return (
            f"Profit Summary:\n"
            f"Total Profit: {metrics['total_profit_formatted']}\n"
            f"Total Revenue: {metrics['total_revenue_formatted']}\n"
            f"Total Cost of Goods Sold: {metrics['total_cogs_formatted']}\n"
            f"Profit Margin: {(metrics['total_profit'] / metrics['total_revenue'] * 100) if metrics['total_revenue'] > 0 else 0:.1f}%"
        )
    
    def _format_quick_summary_text(self):
        """Format the quick summary as text"""
        summary = self.get_quick_summary()
        result = f"Business Performance Summary for {self.data_summary['period']['month']}/{self.data_summary['period']['year']}:\n\n"
        
        for metric, value in summary.items():
            result += f"{metric}: {value}\n"
        
        return result

    def update_data_summary(self, period=None):
        """Update data summary with latest metrics for a specific period"""
        try:
            if self.df is None:
                self._load_data()
                return
                
            if period:
                year, month = period
                
                # Filter for specific period
                if 'Invoice Date' in self.df.columns:
                    df_filtered = self.df[
                        (self.df['Invoice Date'].dt.year == year) & 
                        (self.df['Invoice Date'].dt.month == month)
                    ]
                else:
                    df_filtered = self.df
                
                if len(df_filtered) == 0:
                    print(f"No data for {month}/{year}. Using all available data.")
                    df_filtered = self.df
                
                # Calculate financial metrics
                self._calculate_financial_metrics(df_filtered)
                
                # Calculate product metrics
                self._calculate_product_metrics(df_filtered)
                
                # Calculate customer metrics
                self._calculate_customer_metrics(df_filtered)
                
                # Calculate discount analysis
                self._calculate_discount_metrics(df_filtered)
                
                # Update period information
                self.data_summary["period"] = {
                    "month": month,
                    "year": year
                }
                
                print(f"Data summary updated for {month}/{year}")
                
        except Exception as e:
            print(f"Error updating data summary: {e}")

    def get_metric(self, metric_path):
        """Get specific metric using dot notation (e.g., 'financial_metrics.total_revenue')"""
        try:
            # Split the path into components
            components = metric_path.split('.')
            value = self.data_summary
            for component in components:
                value = value[component]
            return value
        except KeyError:
            return None

    def format_currency(self, value):
        """Format number as currency"""
        return f"${value:,.2f}"

    def format_percentage(self, value):
        """Format number as percentage"""
        return f"{value:.1f}%"

    def get_quick_summary(self):
        """Get a quick summary of key metrics"""
        if not self.data_summary or not self.data_summary.get('financial_metrics'):
            return {"Error": "No data available"}

        metrics = self.data_summary['financial_metrics']
        customer_metrics = self.data_summary.get('customer_metrics', {})
        discount_metrics = self.data_summary.get('discount_analysis', {})
        
        return {
            "Total Revenue": metrics.get('total_revenue_formatted', "$0.00"),
            "Total Profit": metrics.get('total_profit_formatted', "$0.00"),
            "Daily Average Revenue": metrics.get('daily_avg_revenue_formatted', "$0.00"),
            "Average Transaction": metrics.get('avg_transaction_formatted', "$0.00"),
            "Transaction Count": f"{metrics.get('total_transactions', 0):,}",
            "Return Rate": metrics.get('return_rate_formatted', "0.0%"),
            "Customer Repeat Rate": customer_metrics.get('repeat_rate_formatted', "0.0%"),
            "Unique Customers": f"{customer_metrics.get('unique_customers', 0):,}",
            "Average Discount": discount_metrics.get('avg_discount_rate_formatted', "0.0%")
        }
        
    def _get_dataframe(self, year=None, month=None):
        """Tool to get the full dataframe or filtered by period"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
            
        df = self.df.copy()
        
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
            
            if year is not None:
                df = df[df['Invoice Date'].dt.year == year]
                
            if month is not None:
                df = df[df['Invoice Date'].dt.month == month]
        
        return df
    
    def _get_financial_metrics(self):
        """Tool to get financial metrics from data summary"""
        return self.data_summary.get("financial_metrics", {})
    
    def _get_product_metrics(self):
        """Tool to get product metrics from data summary"""
        return self.data_summary.get("product_metrics", {})
    
    def _get_customer_metrics(self):
        """Tool to get customer metrics from data summary"""
        return self.data_summary.get("customer_metrics", {})
    
    def _get_discount_metrics(self):
        """Tool to get discount metrics from data summary"""
        return self.data_summary.get("discount_analysis", {})
    
    def _get_period_info(self):
        """Tool to get period information"""
        return self.data_summary.get("period", {})
    
    @staticmethod
    def get_system_prompt():
        return """You are an expert data analyst specializing in business data summary and analysis.

                Your primary task is to analyze business data and provide clear, accurate summaries and insights.
                When asked about specific metrics, years, or time periods, use the dataframe tools to filter and analyze the data.

                For each query:
                1. Determine what data or metrics are being requested
                2. Use the appropriate tools to access or calculate that information
                3. Present the information clearly with proper formatting
                4. Include relevant context and brief explanations

                If asked about a specific time period or year (like 2022), make sure to filter the data accordingly before analysis.
                When calculating totals, averages, or other metrics, ensure you're using the correctly filtered dataset.

                Always be precise and accurate. If you cannot find or calculate a requested metric, clearly state that 
                and explain why (e.g., "The data for 2022 is not available in the current dataset").
                """

    def get_summary(self, query, context=None):
        """Get summary based on query using AI"""
        try:
            # Use the AI agent to process the query
            enhanced_query = f"""
            Data Summary Request: {query}
            
            Additional Context: {context if context else 'No additional context provided'}
            
            Current Date: {datetime.now().strftime('%Y-%m-%d')}
            """
            
            response = self.agent.run(enhanced_query)
            return response.content if response else "Unable to generate summary."

        except Exception as e:
            print(f"Error getting summary: {str(e)}")
            return f"Error: Unable to generate summary due to technical issues: {str(e)}"