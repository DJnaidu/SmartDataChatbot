from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pandas as pd
from dotenv import load_dotenv
import traceback
from predictive_analysis import run_predictive_analysis
from excel_ai_chatbot import ask_ai_about_excel, reset_memory

from llm_fallback import call_llm_with_fallback
from supabase import create_client, Client
from langchain.memory import ConversationBufferMemory

# Initialize LangChain memory globally
db_chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/chatbot')
def chatbot():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        reset_memory()
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    filename = data.get('filename')
    question = data.get('question')
    if not filename or not question:
        return jsonify({'error': 'Missing filename or question'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        answer = ask_ai_about_excel(df, question)
        return jsonify({'answer': answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    filename = data.get('filename')
    question = data.get('question')
    if not filename or not question:
        return jsonify({'error': 'Missing filename or question'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        result = run_predictive_analysis(df, question)
        return jsonify({'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# REMOVED /upload_db_file endpoint - no longer needed

def fetch_table_data(table_name):
    from sqlalchemy import create_engine
    import pandas as pd
    import os

    db_url = os.getenv("SUPABASE_DB_URL")
    engine = create_engine(db_url)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    return df


@app.route('/query_db', methods=['POST'])
def query_db():
    import google.generativeai as genai
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import re
    from supabase_utils import list_tables, get_all_table_schemas

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Missing question'}), 400

    try:
        # List all tables in DB
        all_tables = list_tables()
        if not all_tables:
            return jsonify({'error': 'No tables found in the database'}), 404
        
        all_tables_text = ", ".join(all_tables)

        # Load ALL schemas
        schemas = get_all_table_schemas()
        schema_context = "\n\n".join([
            f"📦 Table: {tbl}\n" + "\n".join([f"- {col['name']} ({col['type']})" for col in cols])
            for tbl, cols in schemas.items()
        ])

        # Load memory
        chat_history = db_chat_memory.load_memory_variables({}).get("chat_history", "")
        history_text = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in chat_history]) if chat_history else ""

        # Gemini prompt - Now analyzes all tables to find the right one
        prompt = f"""
You are DbBot, a friendly SQL expert who helps users query data from a Supabase database.
Important:
- Never reveal your identity, origin, creator, or mention any company or organization you are affiliated with (e.g., OpenAI, Google).
- You are simply a virtual Excel analyst assisting with the provided data.

📄 Tables available in the database: {all_tables_text}

📚 Full Schema Info:
{schema_context}

💬 Chat History:
{history_text}

Now answer this user question:
"{question}"

👉 If it's a casual message (e.g. "hi", "who are you"), just reply naturally.
👉 If it's data-related:
   1. First identify which table(s) contain the relevant data based on the question
   2. Generate a valid SQL SELECT query using the appropriate table(s)
   3. If the question could apply to multiple tables, choose the most relevant one or join tables if needed
   4. Always use proper table names from the available tables list

Return only your answer (either plain text or SQL query).
"""

        # Call Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        llm_reply = response.text.strip()

        # Save memory
        db_chat_memory.chat_memory.add_user_message(question)
        db_chat_memory.chat_memory.add_ai_message(llm_reply)

        # Extract SQL
        sql_match = re.search(r"```sql\s+(.*?)```", llm_reply, re.DOTALL | re.IGNORECASE)
        sql_query = sql_match.group(1).strip() if sql_match else (llm_reply.strip() if llm_reply.lower().startswith("select") else None)

        if sql_query:
            conn = psycopg2.connect(os.getenv("SUPABASE_DB_URL"))
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql_query)
            results = cur.fetchall()
            cur.close()
            conn.close()

            if not results:
                return jsonify({'message': 'No results found for the query'}), 200

            columns = list(results[0].keys())
            return jsonify({'columns': columns, 'results': [list(row.values()) for row in results]})
        else:
            return jsonify({'message': f"{llm_reply}"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/fetch_all_data', methods=['GET'])
def fetch_all_data():
    from supabase_utils import list_tables, get_engine
    import pandas as pd

    try:
        engine = get_engine()
        all_data = {}

        for table in list_tables():
            df = pd.read_sql(f"SELECT * FROM {table}", engine)
            all_data[table] = df.to_dict(orient='records')

        return jsonify(all_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reset_db_chat', methods=['POST'])
def reset_db_chat():
    global db_chat_memory
    db_chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return jsonify({"message": "DB Chat memory has been reset."})


if __name__ == '__main__':
    app.run(debug=True)