import os
import tempfile
import streamlit as st
from typing import List
import urllib3
import warnings
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import shutil
from functools import lru_cache
from PyPDF2 import PdfReader
import cv2
import numpy as np
import subprocess
import sys

# Set environment variables to control tokenizers behavior
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def check_tesseract():
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True, text=True)
        st.success("Tesseract OCR is installed and accessible.")
    except subprocess.CalledProcessError:
        st.error("Tesseract OCR is not installed or not accessible.")
        st.info("Please install Tesseract OCR and make sure it's in your system PATH.")
        sys.exit(1)

class EnhancedOCRPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path


    def preprocess_image(self, image: Image.Image) -> Image.Image:
        # Convert PIL Image to numpy array
        img = np.array(image)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Noise removal using morphological operations
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # Edge enhancement
        edges = cv2.Canny(opening, 100, 200)
        enhanced = cv2.addWeighted(opening, 1.5, edges, -0.5, 0)
        return Image.fromarray(enhanced)

    def extract_text_from_pdf(self) -> List[Document]:
        texts = []
        pdf = PdfReader(self.file_path)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text.strip():
                texts.append(Document(page_content=text, metadata={"page": page_num, "source": self.file_path}))
        return texts

    def perform_ocr(self) -> List[Document]:
        texts = []
        images = convert_from_path(self.file_path, dpi=300)  # Increased DPI for better quality
        for i, image in enumerate(images, start=1):
            preprocessed_image = self.preprocess_image(image)
            
            # Apply multiple OCR attempts with different configurations
            text = ""
            for config in ['--oem 1 --psm 3', '--oem 1 --psm 6', '--oem 3 --psm 6']:
                text += pytesseract.image_to_string(preprocessed_image, config=config)
            
            if text.strip():
                texts.append(Document(page_content=text, metadata={"page": i, "source": self.file_path}))
            
            if i == 1:
                st.write("Preview of the first page (after preprocessing):")
                st.image(preprocessed_image, caption=f"Page 1 of {os.path.basename(self.file_path)}", use_column_width=True)
        
        return texts

    def load(self) -> List[Document]:
        try:
            texts = self.extract_text_from_pdf()
            
            if not texts or (len(texts) >= 1 and len(texts[0].page_content) < 100):
                st.warning(f"No extractable text found in {os.path.basename(self.file_path)}. Attempting enhanced OCR processing.")
                texts = self.perform_ocr()
            
            self.display_results(texts)
            return texts
        
        except Exception as e:
            st.error(f"Error processing {os.path.basename(self.file_path)}: {str(e)}")
            return []

    def display_results(self, texts: List[Document]):
        if not texts:
            st.error(f"Enhanced OCR processing completed, but no text was extracted from {os.path.basename(self.file_path)}.")
        else:
            st.success(f"Successfully extracted text from {len(texts)} pages in {os.path.basename(self.file_path)}.")
        
        if texts:
            print(f"Number of pages with extracted text: {len(texts)}")
            print(f"Content of the first page:\n{len(texts[0].page_content)}")
            print(f"Content of the documents are :\n{texts[0].page_content}")

class RAGAIEngineer:
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_documents(self, uploaded_files):
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                if file_extension == '.pdf':
                    loader = EnhancedOCRPDFLoader(temp_file_path)
                elif file_extension == '.docx':
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(temp_file_path)
                elif file_extension in ['.xlsx', '.xls']:
                    loader = UnstructuredExcelLoader(temp_file_path)
                else:
                    st.warning(f"Unsupported file format: {uploaded_file.name}")
                    continue

                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = uploaded_file.name

                if loaded_docs:
                    st.success(f"Successfully loaded {len(loaded_docs)} document(s) from {uploaded_file.name}")
                    self.documents.extend(loaded_docs)
                else:
                    st.error(f"No content could be extracted from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(temp_file_path)

        st.write(f"Total documents loaded: {len(self.documents)}")

    def process_documents(self):
        if not self.documents:
            st.error("No documents to process. Please upload documents first.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_documents = text_splitter.split_documents(self.documents)
        st.write(f"Split documents into {len(split_documents)} chunks")

        self.vectorstore = FAISS.from_documents(split_documents, self.embeddings)
        st.write("Vector store created successfully")

    @lru_cache(maxsize=10)
    def get_relevant_documents(self, query: str):
        return self.vectorstore.similarity_search(query, k=3)

    def setup_qa_chain(self):
        if not self.vectorstore:
            st.error("Vector store not created. Please process documents first.")
            return

        try:
            llm = Ollama(model=self.model_name)

            prompt_template = """
            You are an AI assistant specialized in real estate, working for a top real estate agency. 
            Your role is to provide accurate, helpful, and professional information about properties, 
            real estate markets, and related topics based on the documents you've been given.

            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer or if the information isn't in the context provided, 
            say so clearly. Do not make up any information.

            When discussing properties or market trends, be sure to:
            1. Provide specific details from the context when available (e.g., square footage, number of bedrooms, price trends).
            2. Highlight key selling points or unique features of properties.
            3. Explain any real estate terms that might not be familiar to all clients.
            4. If relevant, mention factors that could affect property value or desirability.

            Context: {context}

            Question: {question}

            Please provide a detailed, professional answer as a real estate expert would. 
            Cite relevant parts of the context to support your response. 
            If the context doesn't contain the answer, clearly state that you don't have that specific information.
            """
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            st.write("QA chain set up successfully")
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")

    def chat(self, query: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA chain is not set up. Call setup_qa_chain() first.")

        try:
            relevant_docs = self.get_relevant_documents(query)
            response = self.qa_chain({"query": query, "input_documents": relevant_docs})
            return response['result']
        except Exception as e:
            st.error(f"Error during chat: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."

def main():
    st.set_page_config(page_title="Real Estate AI Assistant", page_icon="🏠")
    st.title("Real Estate AI Assistant powered by Llama 3.12 and Enhanced OCR")

    check_tesseract()  # Check Tesseract installation

    rag_engineer = RAGAIEngineer()

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'xlsx', 'xls'])

    if uploaded_files:
        with st.spinner("Processing documents..."):
            rag_engineer.load_documents(uploaded_files)
            rag_engineer.process_documents()
            rag_engineer.setup_qa_chain()
        st.success("Documents processed successfully!")

    st.subheader("Chat with your Real Estate AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about real estate?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if rag_engineer.qa_chain:
            response = rag_engineer.chat(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Please upload and process documents before chatting.")

if __name__ == "__main__":
    main()