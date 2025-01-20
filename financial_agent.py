from phi.agent import Agent
import streamlit as st
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

# Create basic websearch agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Financial Agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# combining both agents
multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

#multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)


st.header("Financial Agent")
input_text=st.text_input("Enter the Question")
response = multi_ai_agent.print_response(input_text)
submit = st.button("Submit")

if submit:
    st.write(multi_ai_agent.print_response(input_text))