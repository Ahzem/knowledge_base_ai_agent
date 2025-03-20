import logging
import json
from phi.model.google import Gemini
from phi.agent.duckdb import DuckDbAgent

# data_analyst_movies = DuckDbAgent(
#     model=Gemini(model="gemini-2.0-flash"),
#     semantic_model=json.dumps(
#         {
#             "tables": [
#                 {
#                     "name": "movies",
#                     "description": "Contains information about movies from IMDB.",
#                     "path": "./data/IMDB-Movie-Data.csv",
#                 }
#             ]
#         }
#     ),
#     markdown=True,
# )
# data_analyst_movies.print_response(
#     "Show me a histogram of ratings. "
#     "Choose an appropriate bucket size but share how you chose it. "
#     "Show me the result as a pretty ascii diagram",
#     stream=True,
# )data_analyst_movies = DuckDbAgent(
#     model=Gemini(model="gemini-2.0-flash"),
#     semantic_model=json.dumps(
#         {
#             "tables": [
#                 {
#                     "name": "movies",
#                     "description": "Contains information about movies from IMDB.",
#                     "path": "./data/IMDB-Movie-Data.csv",
#                 }
#             ]
#         }
#     ),
#     markdown=True,
# )
# data_analyst_movies.print_response(
#     "Show me a histogram of ratings. "
#     "Choose an appropriate bucket size but share how you chose it. "
#     "Show me the result as a pretty ascii diagram",
#     stream=True,
# )

# data_analyst_sales = DuckDbAgent(
#     model=Gemini(model="gemini-2.0-flash"),
#     semantic_model=json.dumps(
#         {
#             "tables": [
#                 {
#                     "name": "sales",
#                     "description": "Contains retail sales transaction data including product details, pricing, payment methods and returns.",
#                     "path": "./data/MOCK_DATA.csv",
#                 }
#             ]
#         }
#     ),
#     markdown=True,
# )

# # Sample analysis questions
# data_analyst_sales.print_response(
#     """Analyze the sales data and tell me:
#     1. What are the total sales and number of transactions?
#     2. What are the top 5 products by revenue?
#     3. What's the distribution of payment methods?
#     4. What percentage of transactions were returned?
#     Show results with appropriate formatting and any relevant visualizations.""",
#     stream=True,
# )


logging.getLogger('phi').setLevel(logging.WARNING)

def generate_monthly_report(year, month):
    data_analyst_sales = DuckDbAgent(
        model=Gemini(model="gemini-2.0-flash"),
        semantic_model=json.dumps(
            {
                "tables": [
                    {
                        "name": "sales",
                        "description": "Contains retail sales transaction data including product details, pricing, payment methods and returns.",
                        "path": "./data/MOCK_DATA.csv",
                    }
                ]
            }
        ),
        markdown=True,
    )

    query = f"""Please provide ONLY the following report in markdown format, without any explanation or planning:

    # Monthly Sales Report - {month}/{year}

    ## Key Metrics
    - Calculate and show total revenue 
    - Show number of transactions
    - List top 5 selling products by quantity and revenue
    - Show average transaction value
    - Calculate return rate percentage
    - Show most popular payment method

    ## Sales Trends
    - Show daily sales trend as ascii chart

    Note: Format all numbers and currency appropriately. No explanations needed - just the data in markdown format."""

    data_analyst_sales.print_response(query, stream=True)

if __name__ == "__main__":
    try:
        print("\nMonthly Sales Report Generator\n")
        year = int(input("Enter year (2021-2025): "))
        month = int(input("Enter month (1-12): "))
        
        if not (2021 <= year <= 2025) or not (1 <= month <= 12):
            print("\nError: Please enter a valid year (2021-2025) and month (1-12)")
        else:
            print("\nGenerating report...\n")
            generate_monthly_report(year, month)
    except ValueError:
        print("\nError: Please enter valid numeric values")