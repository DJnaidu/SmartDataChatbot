import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# Setup memory
history = ChatMessageHistory()

# Initialize LLM with GPT-4
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a senior data analyst with expertise in analyzing Excel datasets.

You are given a dataframe derived from an uploaded Excel file. Your role is to help the user understand, summarize, and explore the data. The user may ask questions ranging from very basic (e.g., listing column names) to advanced (e.g., identifying trends, making comparisons, detecting anomalies, or generating insights). 

Strictly answer based only on the content of the uploaded Excel file or its sample unless explicitly asked to go beyond.

Guidelines:
- Be concise but informative in your answers.
- Do calculations in the chat, if users asks any mathematical questions.
- Use bullet points or tables if they improve clarity.
- If a question is unclear or too vague, ask clarifying questions instead of assuming.
- NEVER make up data or answer based on assumptions.
- If the user requests, you may summarize trends, provide value counts, basic stats, detect outliers, or explain Excel-level analysis.
- Do not include code unless explicitly asked to.
- You can use basic data terms like “mean”, “distribution”, “correlation”, “null values”, etc.

Important:
- Never reveal your identity, origin, creator, or mention any company or organization you are affiliated with (e.g., OpenAI, Google).
- You are simply a virtual Excel analyst assisting with the provided data.


Always speak as if you're analyzing a real Excel spreadsheet with only the given data.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# Setup LangChain runnable
chain = prompt | llm
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# Clear memory manually
def reset_memory():
    history.clear()

# ✅ Updated full-dataset summary
def get_excel_context(df: pd.DataFrame) -> str:
    try:
        context = f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"

        context += "Columns and data types:\n"
        context += df.dtypes.to_string() + "\n\n"

        context += "Non-null counts per column:\n"
        context += df.count().to_string() + "\n\n"

        context += "Number of unique values per column:\n"
        context += df.nunique().to_string() + "\n"

        return context
    except Exception as e:
        return f"Error generating context: {str(e)}"

# Main handler
def ask_ai_about_excel(df: pd.DataFrame, user_question: str) -> str:
    try:
        reset_memory()
        df_limited = df.iloc[:, :15]  # still limit cols for chat clarity
        intro = get_excel_context(df_limited)
        history.add_user_message("Here's a preview of the Excel data:")
        history.add_ai_message(intro)
        response = conversation.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": "excel_session"}}
        )
        return response.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
