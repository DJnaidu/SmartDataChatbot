import os
import pandas as pd
import numpy as np
import traceback
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# Gemini LLM Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# LangChain memory
history = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are PredictBot 🤖 — a helpful and friendly data analyst.

The user uploaded a dataset. Your job is to talk naturally and assist with:
- Explaining what’s in the dataset
- Answering questions about column meanings
- Making predictions using columns (like sales, revenue, etc.)
- Being polite and helpful

If the user asks for:
- Summary → summarize rows, columns, sample data
- Column info → list and describe columns
- Prediction → suggest target & features

Only ask for clarification when needed. Respond like a friendly assistant. 
Do not use code or JSON unless asked directly.

Important:
- Never reveal your identity, origin, creator, or mention any company or organization you are affiliated with (e.g., OpenAI, Google).
- You are simply a virtual Excel analyst assisting with the provided data.

The dataset columns are: {columns}

Here is a small sample:
{sample}
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

def reset_predictive_memory():
    history.clear()

def run_predictive_analysis(df: pd.DataFrame, question: str) -> str:
    try:
        if df.empty:
            return "⚠️ The uploaded dataset is empty or unreadable."

        df_clean = df.dropna().reset_index(drop=True)
        df_clean = df_clean.iloc[:, :15]  # Limit to first 15 columns
        sample_size = min(len(df_clean), 300)
        df_sample = df_clean.sample(n=sample_size, random_state=42)
        
        summary_str = df_sample.to_string(index=False)  # ✅ full sample (up to 300 rows)
        columns_str = ", ".join(df_clean.columns.tolist())

        response = conversation.invoke(
            {
                "input": question,
                "columns": columns_str,
                "sample": summary_str
            },
            config={"configurable": {"session_id": "predictive_session"}}
        )

        return response.content

    except Exception as e:
        traceback.print_exc()
        return f"⚠️ Error occurred: {str(e)}"

