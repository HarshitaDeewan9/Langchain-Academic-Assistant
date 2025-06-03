import streamlit as st
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import tempfile
from dotenv import load_dotenv
import os

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Smart Academic Assistant", layout="centered")

# -------------------- Title --------------------
st.title("📚 Smart Academic Assistant")
st.write("Upload your academic documents and ask questions to get structured answers.")

# -------------------- File Upload Section --------------------
uploaded_files = st.file_uploader(
    "Upload academic documents (PDF, DOCX, or TXT):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Cache Uploaded Files
file_buffers = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_buffers.append({
            "name": uploaded_file.name,
            "suffix": uploaded_file.name.split(".")[-1],
            "bytes": uploaded_file.read()
        })

# -------------------- Question Input --------------------
question = st.text_input("Enter your academic question:")

# -------------------- Submit Button --------------------
if st.button("Get Answer"):
    if not file_buffers or not question:
        st.warning("Please upload at least one document and enter a question.")
    else:
        try:
            # 1. Load documents using LangChain document loaders
            all_docs = []
            for file in file_buffers:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file['suffix']}") as tmp_file:
                    tmp_file.write(file["bytes"])
                    tmp_path = tmp_file.name

                if file["suffix"] == "pdf":
                    loader = PyPDFLoader(tmp_path)
                elif file["suffix"] == "docx":
                    loader = Docx2txtLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)

                docs = loader.load()
                all_docs.extend(docs)
                os.remove(tmp_path)

            # 2. Split documents using RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)

            # 3. Create embeddings and store in vector store
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)

            # 4. Retrieve relevant chunks based on the question
            retriever = vector_store.as_retriever()

            # 5. Use Groq-hosted LLM via LangChain (e.g., Mixtral, Gemma, Llama3)
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
            llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

            # 6. Use Output Parser to format structured response
            class QAResponse(BaseModel):
                answer: str = Field(description="Answer to the academic question")
                source_document: str = Field(description="Source document of the answer")
                confidence_score: float = Field(description="Estimated confidence score")

            parser = PydanticOutputParser(pydantic_object=QAResponse)

            template = """
                Use the given context to answer the academic question below.

                <context>
                {context}
                </context>

                Question: {question}

                {format_instructions}
            """
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            retrieved_docs = retriever.get_relevant_documents(question)
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])
            formatted_prompt = prompt.format(context=context_text, question=question)

            llm_output = llm.invoke(formatted_prompt)
            parsed_response = parser.parse(str(llm_output.content))

            # Output
            response = {
                "question": question,
                "answer": parsed_response.answer,
                "source_document": parsed_response.source_document,
                "confidence_score": parsed_response.confidence_score
            }

            st.subheader("📄 Answer:")
            st.json(response)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# -------------------- Bonus Section: Agent Tools --------------------
st.markdown("---")
st.subheader("🧠 Bonus Tools ( Optional )")

col1, col2, col3 = st.columns(3)

if file_buffers:
    bonus_documents = []

    for file in file_buffers:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file['suffix']}") as tmp:
            tmp.write(file["bytes"])
            path = tmp.name

        if file["suffix"] == "pdf":
            loader = PyPDFLoader(path)
        elif file["suffix"] == "docx":
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path)

        docs = loader.load()
        bonus_documents.extend(docs)
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    bonus_chunks = splitter.split_documents(bonus_documents)

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

    with col1:
        if st.button("Summarize Document"):
            chain = load_summarize_chain(llm, chain_type="stuff")
            summary = chain.run(bonus_chunks)
            st.info(summary)

    with col2:
        if st.button("Generate MCQs"):
            prompt = PromptTemplate.from_template("""
            Based on the following academic content, generate 5 multiple-choice questions.
            Each question should have 4 options and mark the correct one with an asterisk (*).
            Content:
            {text}
            """)
            chain = prompt | llm
            output = chain.invoke({"text": bonus_chunks[0].page_content})
            st.info(output)

    with col3:
        topic = st.text_input("Enter topic:")
        explain_clicked = st.button("Topic-wise Explanation", key="explain_btn")

        if explain_clicked:
            if topic:
                prompt = PromptTemplate.from_template("""
                Explain the topic \"{topic}\" using the content below.
                Make the explanation student-friendly and include simple examples.
                Content:
                {text}
                """)
                chain = prompt | llm
                output = chain.invoke({"text": bonus_chunks[0].page_content, "topic": topic})
                st.info(output)
            else:
                st.warning("Please enter a topic to explain.")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Mentox Bootcamp · Final Capstone Project · Phase 1")