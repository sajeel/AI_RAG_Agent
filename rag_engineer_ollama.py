# import os
# import tempfile
# import streamlit as st
# from typing import List
# import urllib3
# import warnings
# import easyocr
# from pdf2image import convert_from_path
# from PIL import Image
# import io
# import shutil
# from functools import lru_cache
# from PyPDF2 import PdfReader
# import cv2
# import numpy as np
# import sys

# # Set environment variables to control tokenizers behavior
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Suppress InsecureRequestWarning
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# # Suppress deprecation warnings
# warnings.filterwarnings("ignore", category=DepreconWarning)

# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.chains import RetrievalQA
# # from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQAWithSourcesChain
import os
import tempfile
import streamlit as st
from typing import List
import urllib3
import warnings
import easyocr
import torch
import cv2
import numpy as np
import re

from sklearn.metrics import accuracy_score

from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from functools import lru_cache
 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQAWithSourcesChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
from transformers import AutoTokenizer, AutoModel

from langchain.document_loaders import PDFPlumberLoader
from langchain.schema import Document

# At the beginning of your script, add:
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class EnhancedOCRPDFLoader:
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.reader = easyocr.Reader(['en', 'ar'])  # Initialize EasyOCR with English and Arabic

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

    # def extract_text_from_pdf(self) -> List[Document]:
    #     texts = []
    #     pdf = PdfReader(self.file_path)
    #     # loader = PDFPlumerLoader(self.file_path)
    #     # pages = loader.load_and_split()
    #     for page_num, page in enumerate(pdf.pages, start=1):
    #         text = page.extract_text()
    #         if text.strip():
    #             texts.append(Document(page_content=text, metadata={"page": page_num, "source": self.file_path}))
    #     return texts
    
    def extract_text_from_pdf(self) -> List[Document]:
        loader = PDFPlumberLoader(self.file_path)
        documents = loader.load()
        
        # Add page numbers to metadata
        for i, doc in enumerate(documents, start=1):
            print(f"Content of the documents are :\n{doc.page_content}")
            doc.metadata["page"] = i
        
        
        return documents

    def perform_ocr(self) -> List[Document]:
        texts = []
        images = convert_from_path(self.file_path, dpi=300)  # Increased DPI for better quality
        for i, image in enumerate(images, start=1):
            preprocessed_image = self.preprocess_image(image)
            
            # Perform OCR using EasyOCR
            results = self.reader.readtext(np.array(preprocessed_image))
            
            # Extract text from results
            text = " ".join([result[1] for result in results])
            
            if text.strip():
                texts.append(Document(page_content=text, metadata={"page": i, "source": self.file_path}))
            
            # if i == 1:
#                 st.write("Preview of the first page (after preprocessing):")
#                 st.image(preprocessed_image, caption=f"Page 1 of {os.path.basename(self.file_path)}", use_column_width=True)
        
        return texts

    def load(self) -> List[Document]:
        try:
            texts = self.extract_text_from_pdf()
            st.warning(f"Text extracted from pdf is  {len(texts)}")
            # print(f"Number of pages with extracted text: {len(texts)} and {texts}")
            if not texts or (len(texts) <= 2):
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
        # else:
            # st.success(f"Successfully extracted text from {len(texts)} pages in {os.path.basename(self.file_path)}.")
        
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
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
          # Use a transformer model that supports MPS
        # transformer_model_name = "sentence-transformers/all-MiniLM-L6-v2" 
        transformer_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModel.from_pretrained(transformer_model_name).to(device)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=transformer_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )
        
     

    # Function to generate embeddings for text chunks
    # def generate_embeddings(text_chunks, model_name='nomic-embed-text'):
    #     embeddings = []
    #     for chunk in text_chunks:
    #         # Generate the embedding for each chunk
    #         embedding = ollamalembeddings(model=model_name, prompt=chunk)
    #         embeddings. append (embedding)
    #     return embeddings
    
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
                    # st.success(f"Successfully loaded {len(loaded_docs)} document(s) from {uploaded_file.name}")
                    self.documents.extend(loaded_docs)
                else:
                    st.error(f"No content could be extracted from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(temp_file_path)

        st.write(f"Total documents loaded: {len(self.documents)}")


    # @st.cache_data
    # def _process_documents_cached(self, document_texts: List[str]) -> FAISS:
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=100,
    #         length_function=len,
    #         separators=["\n\n", "\n", " ", ""]
    #     )
    #     split_documents = text_splitter.split_documents([Document(page_content=text) for text in document_texts])
    #     return FAISS.from_documents(split_documents, self.embeddings)

    
    # def process_documents(_self):
    #     if not _self.documents:
    #         st.error("No documents to process. Please upload documents first.")
    #         return

    #     document_texts = [doc.page_content for doc in _self.documents]
    #     _self.vectorstore = _self._process_documents_cached(tuple(document_texts))
    #     st.write(f"Split documents into chunks and created vector store successfully")
        
    # @st.cache_data
    # def _process_documents_cached(_self, document_texts: List[str]) -> FAISS:
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=100,
    #         length_function=len,
    #         separators=["\n\n", "\n", " ", ""]
    #     )
    #     split_documents = text_splitter.split_documents([Document(page_content=text) for text in document_texts])
    #     return FAISS.from_documents(split_documents, _self.embeddings)

    # def process_documents(self):
    #     if not self.documents:
    #         st.error("No documents to process. Please upload documents first.")
    #         return

    #     document_texts = [doc.page_content for doc in self.documents]
    #     self.vectorstore = self._process_documents_cached(document_texts)
    #     st.write(f"Split documents into chunks and created vector store successfully")

    def process_documents(self):
        if not self.documents:
            st.error("No documents to process. Please upload documents first.")
            return

        if 'vectorstore' not in st.session_state:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            # split_documents = text_splitter.split_documents(self.documents)
            split_documents = self.improved_chunking(self.documents)
            st.write(f"Split documents into {len(split_documents)} chunks")

            st.session_state.vectorstore = FAISS.from_documents(split_documents, self.embeddings)
            st.write(f"Split documents into chunks and created vector store successfully")
        else:
            st.write("Using existing vector store")

        self.vectorstore = st.session_state.vectorstore
        return True
    
    def improved_chunking(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # Increased overlap
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(doc_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(doc_chunks)
                    }
                ))
        return chunks
    
    

    # @st.cache_data
    # def process_documents(self):
    #     if not self.documents:
    #         st.error("No documents to process. Please upload documents first.")
    #         return

    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=100,
    #         length_function=len,
    #         separators=["\n\n", "\n", " ", ""]
    #     )
    #     split_documents = text_splitter.split_documents(self.documents)
    #     st.write(f"Split documents into {len(split_documents)} chunks")

    #     self.vectorstore = FAISS.from_documents(split_documents, self.embeddings)
    #     st.write("Vector store created successfully")
         
    #     # from langchain. vectorstores import FAISS
    #     # # Wrap Nvidia texts with their respective metadata into Document objects
    #     # nvidia_documents = [Document(page_content=chunk[ 'text'], metadata=chunk[ 'metadata']) for chunk in nvidia_chunks_with_metadata]
    #     # # Wrap Tesla texts with their respective metadata into Document objects
    #     # tesla_documents = [Document(page_content=chunk['text'], metadata=chunk[ 'metadata']) for chunk in tesla_chunks_with_metadata]
    #     # # Use FAISS instead of Chroma
    #     # nvidia_faiss_store = FAISS. from_documents(documents=nvidia_documents,embedding=nvidia_embeddings)
    #     # tesla_faiss_store = FAISS. from_documents(documents=tesla_documents, embedding=tesla _embeddings)

    # @lru_cache(maxsize=10)
    # def get_relevant_documents(self, query: str):
    #     return self.vectorstore.similarity_search(query, k=5)
    
    @lru_cache(maxsize=10)
    def get_relevant_documents(self, query: str):
         # First-stage retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=10)
        
        # Re-ranking (you can implement a more sophisticated re-ranking method)
        reranked_docs = sorted(initial_docs, key=lambda x: len(x.page_content), reverse=True)[:5]
        
        return reranked_docs

    @st.cache_resource
    def setup_qa_chain(_self):
        if not _self.vectorstore:
            st.error("Vector store not created. Please process documents first.")
            return

        try:
            # llm = Ollama(model=self.model_name)
            llm = Ollama(model=_self.model_name, base_url="http://localhost:11434")

            # prompt_template = """
            # You are an AI assistant specialized in real estate, working for a top real estate agency. 
			# Your role is to provide accurate, helpful, and professional information about properties, 
			# real estate markets, and related topics based on the documents you've been given. 
            
            # you should be able to calculate, analysis data and calculate statistic values,
            # compute projected values , apply deduction logic as well. 

			# Use the following pieces of context to answer the question at the end. 
			# If you don't know the answer or if the information isn't in the context provided, 
			# say so clearly. Do not make up any information.

			# When discussing properties or market trends, be sure to:
			# 1. Provide specific details from the context when available (e.g., square footage, number of bedrooms, price trends, monthly rents, annual rent).
			# 2. Highlight key selling points or unique features of properties.
			# 3. Explain any real estate terms that might not be familiar to all clients.
			# 4. If relevant, mention factors that could affect property value or desirability.
			# 5. Offer general advice on real estate practices and procedures.
			# 6. Discuss common real estate scenarios and how they are typically handled.
            # 7. Perform mathematically calculations based on the information available. 
            # 8. Give projection values, able to compute, do statistical analysis as well. 

			# Remember, your goal is to be informative and helpful. do not focus on legal considerations, 
			# focus on providing useful information rather than giving specific legal advice. If a question touches on complex 
			# legal matters, suggest what you can extract from document.

			# Context: {context}

			# Question: {question}

			# Please provide a detailed, professional answer as a real estate expert would. 
			# Cite relevant parts of the context to support your response. 
			# If the context doesn't contain the answer, clearly state that you don't have that specific information.
			# Aim to be helpful and informative while maintaining professional ethics.
            # """
            
            
            prompt_template = """
            You are an AI assistant specialized in real estate, working for a top real estate agency. 
            Your role is to provide accurate, helpful, and professional information about properties, 
            real estate markets, and related topics based on the documents you've been given.

            To answer the question, follow these steps:
            1. Identify the key information needed from the context.
            2. Extract relevant data points from the provided documents.
            3. If calculations are needed, clearly show your work step-by-step.
            4. Apply logical deductions based on the information available.
            5. Synthesize the information to form a comprehensive answer.

            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer or if the information isn't in the context provided, 
            say so clearly. Do not make up any information.

            Context: {context}

            Question: {question}

            Please provide a detailed, professional answer as a real estate expert would. 
            Show your reasoning and calculations clearly. 
            Cite relevant parts of the context to support your response. 
            If the context doesn't contain the answer, clearly state that you don't have that specific information.
            """
            
            
            # PROMPT = PromptTemplate(
            #     # template=prompt_template, input_variables=["context", "question"]
            #     template=prompt_template, input_variables=["summaries", "question"]
                
            # )
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "numerical_context", "question"]
            )


            # Create an LLM chain
            llm_chain = LLMChain(llm=llm, prompt=PROMPT)

            # Create a StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context",
                document_prompt=PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}"
                )
            )

			

            # Create the retrieval chain
            _self.qa_chain = RetrievalQAWithSourcesChain(
                combine_documents_chain=stuff_chain,
                retriever = _self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}),
#                 retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
            
            # Store the QA chain in session state
            st.session_state.qa_chain = _self.qa_chain

#           self.qa_chain = RetrievalQA.from_chain_type(
            # self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            #     llm=llm,
            #     chain_type="stuff",
            #     retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            #     chain_type_kwargs={
            #         "prompt": PROMPT,
            #         "document_variable_name": "summaries"
            #     },
            #     return_source_documents=True
            # )
            # self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            #     llm=llm,
            #     chain_type="stuff",
            #     retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            #     chain_type_kwargs={"prompt": PROMPT}

            # )
            
            
            st.write("QA chain set up successfully")
            return True
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            return False

    
    def extract_numbers(self, text):
        return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]
    
    def numerical_operations(self, numbers):
        if not numbers:
            return {}
        return {
            "sum": np.sum(numbers),
            "average": np.mean(numbers),
            "median": np.median(numbers),
            "min": np.min(numbers),
            "max": np.max(numbers)
        }
            
    def chat(self, query: str) -> str:
        
         # Retrieve the QA chain from session state
        self.qa_chain = st.session_state.get('qa_chain')
        
        if not self.qa_chain:
            raise ValueError("QA chain is not set up. Call setup_qa_chain() first.")
        try:
            relevant_docs = self.get_relevant_documents(query)
            
            # # Extract numbers from relevant documents
            # all_numbers = []
            # for doc in relevant_docs:
            #     all_numbers.extend(self.extract_numbers(doc.page_content))
            
            # # Perform numerical operations
            # numerical_results = self.numerical_operations(all_numbers)
            
            # # Add numerical results to the context
            # numerical_context = f"Numerical analysis of relevant data: {numerical_results}\n\n"
            
            
            # # Combine the query with numerical context
            # enhanced_query = f"{query}\n\nNumerical Context: {numerical_context}"
            
            # # Call the QA chain with the enhanced query
            # response = self.qa_chain({"question": enhanced_query})
            
            
            # response = self.qa_chain({
            #     "question": query, 
            #     "input_documents": relevant_docs
            #     # "numerical_context": numerical_context
            # })
            
            
            # answer = response['answer']
            # sources = set([doc.metadata.get('source', 'Unknown source') for doc in response['source_documents']])
            
            # formatted_response = f"{answer} \n\nSources:\n" + "\n".join(sources)
            # return formatted_response
        
            response = self.qa_chain({"question": query})
            answer = response.get('answer', "I couldn't generate an answer.")
            
            # Check if source_documents exists and has items
            if 'source_documents' in response and response['source_documents']:
                sources = set([doc.metadata.get('source', 'Unknown source') for doc in response['source_documents']])
                formatted_response = f"{answer}\n\nSources:\n" + "\n".join(sources)
            else:
                formatted_response = f"{answer}\n\nNo source documents were returned."
            
            return formatted_response
        except Exception as e:
            st.error(f"Error during chat: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."

#     def chat(self, query: str) -> tuple:
#         if not self.qa_chain:
#             raise ValueError("QA chain is not set up. Call setup_qa_chain() first.")
#         try:
           
#             response = self.qa_chain({"question": query})
#             answer = response['answer']
#             sources = set([doc.metadata['source'] for doc in response['source_documents']])

#             # accuracy = self.calculate_accuracy(answer, correct_answer)

            
#             # Format the response with source information
#             formatted_response = f"{answer}\n\nSources:\n" + "\n".join(sources)
#             return formatted_response
#             # return formatted_response, accuracy
            
#             # relevant_docs = self.get_relevant_documents(query)
# #             response = self.qa_chain({"query": query, "input_documents": relevant_docs})
# #             return response['result']
#         except Exception as e:
#             st.error(f"Error during chat: {str(e)}")
#             return "I'm sorry, I encountered an error while processing your question.", 0.0
        


    def calculate_accuracy(self, generated_answer: str, correct_answer: str) -> float:
        print(f"Generated answer: {generated_answer}")
        print(f"Correct answer: {correct_answer}")
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([generated_answer, correct_answer])
        print(f"Vectors shape: {vectors.shape}")
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        print(f"Calculated similarity: {similarity}")
        return similarity

    # def chat(self, query: str, correct_answer: str = None) -> tuple:
    #     if not self.qa_chain:
    #         raise ValueError("QA chain is not set up. Call setup_qa_chain() first.")
    #     try:
    #         response = self.qa_chain({"question": query})
    #         answer = response['answer']
    #         sources = set([doc.metadata['source'] for doc in response['source_documents']])
            
    #         # Format the response with source information
    #         formatted_response = f"{answer}\n\nSources:\n" + "\n".join(sources)
            
    #         accuracy = None
    #         if correct_answer:
    #             accuracy = self.calculate_accuracy(answer, correct_answer)
            
    #         return formatted_response, accuracy
    #     except Exception as e:
    #         st.error(f"Error during chat: {str(e)}")
    #         return "I'm sorry, I encountered an error while processing your question.", None

@st.cache_resource
def load_model():
    return RAGAIEngineer()

def main():
    st.set_page_config(page_title="Real Estate AI Assistant", page_icon="🏠")
    st.title("Real Estate AI Assistant powered by Llama 3.12 and Enhanced OCR")

    if 'device' not in st.session_state:
        st.session_state.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        st.write(f"Using device: {st.session_state.device}")
    # rag_engineer = RAGAIEngineer()
    rag_engineer = load_model()
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

    if not st.session_state.documents_processed:
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'xlsx', 'xls'])
        if uploaded_files:
            with st.spinner("Processing documents..."):
                rag_engineer.load_documents(uploaded_files)
                if rag_engineer.process_documents():
                    if rag_engineer.setup_qa_chain():
                        st.session_state.documents_processed = True
                        st.success("Documents processed successfully!")
                    else:
                        st.error("Failed to set up QA chain. Please try again.")
                else:
                    st.error("Failed to process documents. Please try again.")

    # if uploaded_files:
    #     with st.spinner("Processing documents..."):
    #         rag_engineer.load_documents(uploaded_files)
    #         rag_engineer.process_documents()
    #         rag_engineer.setup_qa_chain()
    #     st.success("Documents processed successfully!")

    st.subheader("Chat with your Real Estate AI Assistant")
    
    if st.session_state.documents_processed:
        st.subheader("Chat with your Real Estate AI Assistant")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What would you like to know about real estate?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = rag_engineer.chat(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and process documents before chatting.")
        

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # if prompt := st.chat_input("What would you like to know about real estate?"):
    #     st.chat_message("user").markdown(prompt)
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    #     if rag_engineer.qa_chain:
    #         response = rag_engineer.chat(prompt)
    #         with st.chat_message("assistant"):
    #             st.markdown(response)
    #         st.session_state.messages.append({"role": "assistant", "content": response})
    #     else:
    #         st.error("Please upload and process documents before chatting.")
            
    
    

if __name__ == "__main__":
    main()