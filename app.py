import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Load Vector Store ---
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local("index_store", embedding, allow_dangerous_deserialization=True)

# --- Load LLM (GPT-Neo or compatible) ---
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=generator)

# --- Ask function ---
def ask(user_query):
    # Search similar docs from vector store
    docs = vectorstore.similarity_search(user_query, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # Extract lines starting with Loan_ID: and prepend header for CSV parsing
    data_lines = ["Loan_ID," + line for line in context.split("Loan_ID:")[1:]]

    # Parse each line into dictionary safely
    dict_list = []
    for line in data_lines:
        item_dict = {}
        # Split by ', ' to get key-value pairs
        for item in line.split(", "):
            if ": " in item:
                key, value = item.split(": ", 1)
                item_dict[key.strip()] = value.strip()
            else:
                # Handle cases where ':' is missing - maybe just a header or malformed
                # Try splitting by ',' if possible or skip
                parts = item.split(",")
                if len(parts) == 2:
                    item_dict[parts[0].strip()] = parts[1].strip()
                else:
                    # Could log or ignore
                    pass
        if item_dict:
            dict_list.append(item_dict)

    # Convert to DataFrame
    df = pd.DataFrame(dict_list)

    # Convert numeric columns safely
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Make sure Loan_Status column exists
    if 'Loan_Status' not in df.columns:
        st.warning("Loan_Status column missing in data.")
        return "No valid loan status data found.", None

    # Plot boxplot
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Loan_Status', y='ApplicantIncome', ax=ax)
    ax.set_title("Applicant Income by Loan Status")

    # Save plot to buffer
    plot_buf = io.BytesIO()
    fig.savefig(plot_buf, format='png')
    plot_buf.seek(0)
    plt.close(fig)  # Close plot to free memory

    # Summarize loan approvals
    accepted = df[df['Loan_Status'] == 'Y'].shape[0]
    rejected = df[df['Loan_Status'] == 'N'].shape[0]
    summary = (
        f"In the retrieved loan data, {accepted} applications were approved and "
        f"{rejected} were rejected. Income levels seem to vary across status groups. "
        f"Here's the breakdown plotted above."
    )

    # Prepare prompt for LLM
    prompt = (
        f"You are a financial expert in loan analysis.\n"
        f"Context summary: {summary}\n\n"
        f"Question: {user_query}\n"
        f"Answer in detail:"
    )

    result = generator(
        prompt,
        max_new_tokens=300,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=50256,
    )

    answer = result[0]["generated_text"].replace(prompt, '').strip()
    return answer, plot_buf

# --- Streamlit UI ---
st.set_page_config(page_title="Loan Chatbot", layout="centered")
st.title("Loan Approval Q&A Chatbot")
st.write("Ask any question about loan approvals based on training data context.")

user_query = st.text_input("Enter your question:")

if user_query:
    with st.spinner("Thinking..."):
        answer, plot_buf = ask(user_query)
        if plot_buf:
            st.markdown("### Visual Insight")
            st.image(plot_buf, caption="Applicant Income vs Loan Status", use_column_width=True)

        st.markdown("### Answer")
        st.success(answer)
