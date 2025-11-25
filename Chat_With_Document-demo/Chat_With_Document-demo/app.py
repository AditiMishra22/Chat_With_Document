from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time
from werkzeug.utils import secure_filename
import uuid
import asyncio
import nest_asyncio

# Apply nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploaded_documents'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.getenv('SECRET_KEY', str(uuid.uuid4()))

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize API keys
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# HuggingFace free local embeddings (NO API KEY NEEDED)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
     You are an AI assistant answering questions based ONLY on the context provided.  
    - If the answer is not in the context, reply: "I could not find this information in the documents."  
    - Be concise and factual.  
    
    Context:
    {context}
    
    Question: {input}
    """
)

# File extension check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure event loop exists
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Logging
def flash_and_print(message):
    flash(message)
    print(message)

# Process uploaded files
def process_documents(files):
    loop = get_or_create_event_loop()

    total_files = len(files)
    processed_files = 0
    total_pages = 0

    all_docs = []

    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                flash_and_print(f"Processing {filename}...")

                loader = PyPDFLoader(filepath)
                docs = loader.load()

                file_pages = len(docs)
                total_pages += file_pages

                all_docs.extend(docs)
                processed_files += 1

                flash_and_print(f"Loaded {filename} ({file_pages} pages)")
            except Exception as e:
                flash_and_print(f"Error processing {file.filename}: {str(e)}")

    if not all_docs:
        flash_and_print("No documents were successfully processed!")
        return None

    flash_and_print(f"Processed {processed_files}/{total_files} files, {total_pages} pages")

    # Split documents
    flash_and_print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    final_documents = text_splitter.split_documents(all_docs)

    total_chunks = len(final_documents)
    flash_and_print(f"Created {total_chunks} document chunks")

    if total_chunks == 0:
        flash_and_print("No chunks created")
        return None

    # Vector store using HuggingFace embeddings
    flash_and_print("Creating FAISS vector store...")
    vectors = FAISS.from_documents(final_documents, embeddings)

    flash_and_print("Vector index created successfully")
    return vectors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files[]')

    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)

    try:
        vectors = process_documents(files)
        if vectors is not None:
            session['has_vectors'] = True
            app.config['vectors'] = vectors
            flash('Documents processed successfully!')
        else:
            flash('Failed to create vector embeddings')
    except Exception as e:
        flash(f'Error processing documents: {str(e)}')

    return redirect(url_for('index'))

@app.route('/query', methods=['POST'])
def query_documents():
    loop = get_or_create_event_loop()

    if not session.get('has_vectors', False):
        return jsonify({'error': 'No documents processed yet'})

    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'No query provided'})

    try:
        vectors = app.config.get('vectors')

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever(search_kwargs={"k": 6})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': query})
        response_time = time.process_time() - start

        context_snippets = [doc.page_content for doc in response.get("context", [])]

        return jsonify({
            'answer': response['answer'],
            'response_time': response_time,
            'context_snippets': context_snippets
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)