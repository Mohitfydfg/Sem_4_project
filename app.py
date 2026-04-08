import streamlit as st
from sqlalchemy import create_engine, text
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="University AI Assistant", layout="wide")

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
@st.cache_resource
def get_engine():
    return create_engine(
        "mysql+mysqlconnector://root:password@localhost/university_ai"
    )

engine = get_engine()
db = SQLDatabase(engine)

# -----------------------------
# LLM SETUP
# -----------------------------
@st.cache_resource
def get_llm():
    return Ollama(model="llama3")

llm = get_llm()

# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are a university database assistant.

Rules:
- NEVER greet the user
- NEVER explain SQL
- Return ONLY the final answer
"""

# -----------------------------
# CREATE SQL AGENT
# -----------------------------
@st.cache_resource
def get_agent():
    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=False,
        return_direct=True,
        handle_parsing_errors=True
    )

agent_executor = get_agent()

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🎓 LJ University AI Assistant")
st.markdown("Ask about faculties, scholarships, departments, etc.")

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.text_input("Ask your question:")

# -----------------------------
# SIDEBAR (SCHOLARSHIP FILTER)
# -----------------------------
st.sidebar.title("User Details")

caste = st.sidebar.selectbox("Select Your Caste", ["SC", "OBC", "General"])
income = st.sidebar.number_input("Enter Annual Income", min_value=0)

if st.sidebar.button("Find Eligible Scholarships"):
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT name, caste, income_limit 
                FROM scholarships 
                WHERE caste = :caste 
                AND income_limit >= :income
            """)
            result = conn.execute(query, {"caste": caste, "income": income})
            rows = result.fetchall()

            if rows:
                st.sidebar.success("Eligible Scholarships:")
                for row in rows:
                    st.sidebar.write(f"• {row.name} (Limit: ₹{row.income_limit})")
            else:
                st.sidebar.warning("No scholarships found.")

    except Exception as e:
        st.sidebar.error(f"Database Error: {e}")

# -----------------------------
# MAIN QUERY BUTTON
# -----------------------------
if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                final_input = f"{SYSTEM_PROMPT}\nQuestion: {user_input}"

                result = agent_executor.invoke({"input": final_input})

                # Handle output cleanly
                if isinstance(result, dict):
                    output = result.get("output", "No result found")
                else:
                    output = str(result)

                st.success(output)

            except Exception as e:
                st.error(f"Error: {e}")