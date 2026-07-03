import os
import shutil
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from chatbot.embeddings import get_embedding_model

if os.environ.get("VERCEL"):
    DB_FAISS_PATH = "/tmp/faiss_index"
else:
    DB_FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faiss_index")

class MockLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # Generate mock JSON for quizzes, flashcards, notes, comparisons, and interview prep
        if "JSON" in prompt and "quiz" in prompt.lower():
            return '[{"question": "How do you enable full AI Mode on Vercel?", "options": ["Set HF_TOKEN env var", "Set OPENAI_API_KEY env var", "Both work", "Do nothing"], "answer": "Both work", "difficulty": "Medium"}]'
        elif "JSON" in prompt and "flashcard" in prompt.lower():
            return '{"flashcards": [{"front": "Vercel Key Configuration", "back": "Define HF_TOKEN or OPENAI_API_KEY under Settings > Environment Variables in the Vercel dashboard."}]}'
        elif "JSON" in prompt and "notes" in prompt.lower():
            return '{"short_notes": "### DocuMind is in Demo Mode\\n\\nPlease configure HF_TOKEN in Vercel to activate.", "bullet_points": "- Missing credentials\\n- Configure HF_TOKEN\\n- Configure OPENAI_API_KEY", "concepts": "- **Demo Mode**: Fallback state when keys are absent.\\n- **Production Mode**: Full RAG pipeline.", "revision": "Check Vercel Environment Variables.", "exam_prep": "Q: How to fix 500 error?\\nA: By wrapping the initialization in try-except."}'
        elif "JSON" in prompt and "compare" in prompt.lower():
            return '{"purpose": {"doc1": "Demo", "doc2": "Demo", "comparison": "Both are running in Demo Mode."}, "topics": {"doc1": "Demo", "doc2": "Demo", "comparison": "Both are running in Demo Mode."}, "concepts": {"doc1": "Demo", "doc2": "Demo", "comparison": "Both are running in Demo Mode."}, "differences": "None", "similarities": "Both are running in Demo Mode.", "conclusion": "Please configure HF_TOKEN."}'
        elif "JSON" in prompt and "interview" in prompt.lower():
            return '{"questions": [{"question": "What happens if HF_TOKEN is missing?", "answer": "The application falls back to Demo Mode.", "difficulty": "Basic", "topic": "Vercel Deployment"}]}'
            
        return (
            "⚠️ **DocuMind is running in Demo Mode** because no cloud AI credentials are configured in Vercel.\n\n"
            "To get actual answers from the AI based on your PDF, please add your **`HF_TOKEN`** or **`OPENAI_API_KEY`** "
            "to your environment variables in your **Vercel Project Dashboard** (under Settings > Environment Variables) "
            "and redeploy the application.\n\n"
            "Here is the text retrieved from your document matching your query:\n\n"
            f"{prompt[:500]}..."
        )

    @property
    def _llm_type(self) -> str:
        return "demo"

class RAGPipeline:
    def __init__(self, model_name: str = "llama3.2"):
        self.embeddings = get_embedding_model()
        self.model_name = model_name
        self.vector_store = self._load_vector_store()
        
        # Initialize LLM based on environment
        self.llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        
        # Override to cloud on Vercel if provider is still set to local ollama
        if os.environ.get("VERCEL") and self.llm_provider == "ollama":
            if os.environ.get("GROQ_API_KEY"):
                self.llm_provider = "groq"
            elif os.environ.get("OPENAI_API_KEY"):
                self.llm_provider = "openai"
            elif os.environ.get("HF_TOKEN"):
                self.llm_provider = "huggingface"
            else:
                self.llm_provider = "huggingface"  # Fallback
                
        if self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0.1
            )
            self.json_llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        elif self.llm_provider == "groq":
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("GROQ_API_KEY")
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                model="llama3-8b-8192",
                temperature=0.1
            )
            self.json_llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                model="llama3-8b-8192",
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        elif self.llm_provider == "huggingface":
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                # Fallback to MockLLM on Vercel instead of crashing
                self.llm = MockLLM()
                self.json_llm = MockLLM()
            else:
                from langchain_community.llms import HuggingFaceEndpoint
                self.llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.2-3B-Instruct",
                    huggingfacehub_api_token=hf_token,
                    temperature=0.1
                )
                self.json_llm = self.llm
        else: # Default local Ollama
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=0.1,
            )
            self.json_llm = OllamaLLM(
                model=self.model_name,
                temperature=0.1,
                format="json"
            )

    def _invoke_llm(self, llm_instance, prompt_or_messages) -> str:
        """
        Helper method to invoke the LLM correctly whether it's a Chat Model or a Completion Model.
        """
        if hasattr(llm_instance, "invoke") and ("Chat" in llm_instance.__class__.__name__):
            msgs = prompt_or_messages
            if isinstance(prompt_or_messages, str):
                msgs = [("user", prompt_or_messages)]
            return llm_instance.invoke(msgs).content
        else:
            if isinstance(prompt_or_messages, list):
                prompt_template = ChatPromptTemplate.from_messages(prompt_or_messages)
                prompt_or_messages = prompt_template.format()
            return llm_instance.invoke(prompt_or_messages)

    def _stream_llm(self, llm_instance, prompt_or_messages):
        """
        Helper method to stream from the LLM correctly whether it's a Chat Model or a Completion Model.
        """
        if hasattr(llm_instance, "stream") and ("Chat" in llm_instance.__class__.__name__):
            msgs = prompt_or_messages
            if isinstance(prompt_or_messages, str):
                msgs = [("user", prompt_or_messages)]
            for chunk in llm_instance.stream(msgs):
                yield chunk.content
        else:
            if isinstance(prompt_or_messages, list):
                prompt_template = ChatPromptTemplate.from_messages(prompt_or_messages)
                prompt_or_messages = prompt_template.format()
            for chunk in llm_instance.stream(prompt_or_messages):
                yield chunk
        
    def _load_vector_store(self) -> Optional[FAISS]:
        """
        Loads the local FAISS index if it exists.
        """
        if os.path.exists(DB_FAISS_PATH):
            try:
                return FAISS.load_local(
                    DB_FAISS_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                return None
        return None

    def add_documents(self, documents: List[Any]) -> None:
        """
        Adds text chunks to the FAISS vector store and saves it locally.
        """
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        # Save the vector store locally
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        self.vector_store.save_local(DB_FAISS_PATH)

    def clear_database(self) -> None:
        """
        Clears the local FAISS vector store and deletes the index directory.
        """
        self.vector_store = None
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH, ignore_errors=True)

    def rephrase_question(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Rephrases a follow-up question to be a standalone question based on history.
        """
        if not chat_history:
            return question
            
        history_str = ""
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['text']}\n"
            
        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone question (meaning it can be understood "
            "without the conversation history). Do NOT answer the question. Just return the rephrased question "
            "and nothing else.\n\n"
            f"Chat History:\n{history_str}\n"
            f"Follow-up Question: {question}\n\n"
            "Standalone Question:"
        )
        try:
            standalone = self._invoke_llm(self.llm, prompt).strip()
            if standalone.startswith("Standalone Question:"):
                standalone = standalone.replace("Standalone Question:", "").strip()
            return standalone
        except Exception as e:
            print(f"Error rephrasing question: {e}")
            return question

    def answer_question(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Retrieves context from the FAISS vector store and answers the question
        using the local Ollama LLM.
        """
        if chat_history is None:
            chat_history = []
            
        if self.vector_store is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload a PDF file to begin.",
                "sources": []
            }
            
        rephrased_query = self.rephrase_question(question, chat_history)
        docs_and_scores = self.vector_store.similarity_search_with_score(rephrased_query, k=4)
        
        if not docs_and_scores:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": []
            }
            
        context_parts = []
        sources = []
        for doc, score in docs_and_scores:
            similarity = 1.0 - (score / 2.0)
            confidence = max(0, min(100, int(similarity * 100)))
            page = doc.metadata.get("page", 0) + 1
            filename = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
            chunk_index = doc.metadata.get("chunk_index", 1)
            
            source_info = {
                "page": page,
                "source": filename,
                "chunk_index": chunk_index,
                "confidence": confidence,
                "snippet": doc.page_content.strip()
            }
            sources.append(source_info)
            context_parts.append(f"[Source: {filename}, Page: {page}, Chunk: {chunk_index}]\n{doc.page_content}")
            
        context_str = "\n\n".join(context_parts)
        
        system_prompt = (
            "You are a helpful, professional AI assistant. Use the following pieces of retrieved "
            "context from the uploaded PDF documents to answer the question.\n"
            "Answer the question ONLY using the provided context. If the answer cannot be found in the context, "
            "reply exactly with: \"I couldn't find this information in the uploaded document.\"\n"
            "Do not make up, extrapolate, or assume any information. Keep the answer factual and directly based on the context.\n\n"
            f"Context:\n{context_str}"
        )
        
        messages = [("system", system_prompt)]
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append((role, msg["text"]))
        messages.append(("human", question))
        
        prompt_template = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = prompt_template.format()
        
        try:
            answer = self._invoke_llm(self.llm, messages)
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error communicating with LLM: {str(e)}",
                "sources": []
            }

    def answer_question_stream(self, question: str, chat_history: List[Dict[str, str]]):
        """
        Generator that yields status updates, token chunks, and final sources.
        """
        if self.vector_store is None:
            yield {"type": "token", "content": "No documents have been uploaded yet. Please upload a PDF file to begin."}
            return

        yield {"type": "status", "content": "Searching documents..."}
        
        # 1. Rephrase the question
        rephrased_query = self.rephrase_question(question, chat_history)
        
        # 2. Retrieve documents
        docs_and_scores = self.vector_store.similarity_search_with_score(rephrased_query, k=4)
        
        if not docs_and_scores:
            yield {"type": "status", "content": "Analyzing context..."}
            yield {"type": "status", "content": "Generating response..."}
            yield {"type": "token", "content": "I couldn't find this information in the uploaded document."}
            return

        yield {"type": "status", "content": "Analyzing context..."}
        
        context_parts = []
        sources = []
        for doc, score in docs_and_scores:
            similarity = 1.0 - (score / 2.0)
            confidence = max(0, min(100, int(similarity * 100)))
            
            page = doc.metadata.get("page", 0) + 1
            filename = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
            chunk_index = doc.metadata.get("chunk_index", 1)
            
            source_info = {
                "page": page,
                "source": filename,
                "chunk_index": chunk_index,
                "confidence": confidence,
                "snippet": doc.page_content.strip()
            }
            sources.append(source_info)
            context_parts.append(f"[Source: {filename}, Page: {page}, Chunk: {chunk_index}]\n{doc.page_content}")

        context_str = "\n\n".join(context_parts)
        
        yield {"type": "status", "content": "Generating response..."}
        
        # 3. Formulate the system prompt
        system_prompt = (
            "You are a helpful, professional AI assistant. Use the following pieces of retrieved "
            "context from the uploaded PDF documents to answer the question.\n"
            "Answer the question ONLY using the provided context. If the answer cannot be found in the context, "
            "reply exactly with: \"I couldn't find this information in the uploaded document.\"\n"
            "Do not make up, extrapolate, or assume any information. Keep the answer factual and directly based on the context.\n\n"
            f"Context:\n{context_str}"
        )
        
        messages = [("system", system_prompt)]
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append((role, msg["text"]))
        messages.append(("human", question))
        
        prompt_template = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = prompt_template.format()
        
        # 4. Stream tokens
        try:
            for chunk in self._stream_llm(self.llm, messages):
                yield {"type": "token", "content": chunk}
        except Exception as e:
            yield {
                "type": "token",
                "content": f"Error communicating with LLM: {str(e)}."
            }
            
        # 5. Yield sources
        yield {"type": "sources", "content": sources}

    def summarize_pdf(self, filename: str) -> str:
        """
        Retrieves all chunks for a specific PDF, sorts them, and uses the local LLM 
        to generate a structured summary.
        """
        if self.vector_store is None:
            return "No documents have been uploaded yet."
            
        # Retrieve all chunks for this PDF
        pdf_chunks = []
        for doc in self.vector_store.docstore._dict.values():
            source = doc.metadata.get("source", "")
            if os.path.basename(source) == filename:
                pdf_chunks.append(doc)
                
        if not pdf_chunks:
            return f"No text chunks found for document '{filename}'."
            
        # Sort chunks by chunk_index to maintain original order
        pdf_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text
        full_text = "\n\n".join([doc.page_content for doc in pdf_chunks])
        
        # Limit text length to prevent context overflow (approx 6,000 words)
        if len(full_text) > 30000:
            full_text = full_text[:30000] + "\n\n[Text truncated for summarization...]"
            
        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert document analyzer. Analyze the provided document text and generate a structured, professional summary.\n"
                "The summary MUST follow this exact format:\n\n"
                "### 1. Executive Summary\n"
                "[Provide a concise 2-3 sentence overview of the document's main purpose and scope.]\n\n"
                "### 2. Key Points\n"
                "- [Key point 1]\n"
                "- [Key point 2]\n"
                "- [Key point 3]\n"
                "...\n\n"
                "### 3. Important Terms\n"
                "- **[Term 1]**: [Brief definition/context from the text]\n"
                "- **[Term 2]**: [Brief definition/context from the text]\n"
                "...\n\n"
                "### 4. Conclusion\n"
                "[Provide a brief concluding statement summarizing the ultimate takeaway of the document.]\n\n"
                "Be direct, factual, and do not include any introductory or concluding conversational filler."
            )),
            ("human", "Document Text:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(text=full_text)
        
        try:
            summary = self._invoke_llm(self.llm, formatted_prompt)
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_quiz(self, filename: str) -> str:
        """
        Retrieves all chunks for a specific PDF, sorts them, and uses the local LLM 
        to generate a JSON array of exactly 10 multiple-choice questions.
        """
        if self.vector_store is None:
            return "[]"
            
        # Retrieve all chunks for this PDF
        pdf_chunks = []
        for doc in self.vector_store.docstore._dict.values():
            source = doc.metadata.get("source", "")
            if os.path.basename(source) == filename:
                pdf_chunks.append(doc)
                
        if not pdf_chunks:
            return "[]"
            
        # Sort chunks
        pdf_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text (up to 30,000 characters to prevent overflow)
        full_text = "\n\n".join([doc.page_content for doc in pdf_chunks])
        if len(full_text) > 30000:
            full_text = full_text[:30000]
            
        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert educator. Analyze the provided document text and generate a multiple-choice quiz of exactly 10 questions.\n"
                "The output MUST be a valid JSON array of objects. Do NOT include any markdown formatting (like ```json or ```), introductory text, or explanatory text. Output ONLY the raw JSON.\n\n"
                "Each object in the JSON array must have the following keys:\n"
                "1. \"question\": The question text.\n"
                "2. \"options\": An array of exactly 4 strings representing the multiple-choice options.\n"
                "3. \"answer\": The correct option (must match one of the strings in the \"options\" array exactly).\n"
                "4. \"difficulty\": The difficulty level of the question. Must be one of: \"Easy\", \"Medium\", or \"Hard\".\n\n"
                "Distribute the difficulty levels across the 10 questions (e.g., 3 Easy, 4 Medium, 3 Hard)."
            )),
            ("human", "Document Text:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(text=full_text)
        
        try:
            quiz_output = self._invoke_llm(self.json_llm, formatted_prompt).strip()
            
            # Clean up markdown code blocks if the LLM included them
            if quiz_output.startswith("```"):
                lines = quiz_output.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                quiz_output = "\n".join(lines).strip()
                
            return quiz_output
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return "[]"

    def explain_simply(self, text: str) -> str:
        """
        Rewrites the provided text using very simple English suitable for a 10-year-old child.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful teacher. Rewrite the provided text so that a 10-year-old child can easily understand it.\n"
                "Follow these strict rules:\n"
                "1. Use very simple English and extremely short, punchy sentences.\n"
                "2. Avoid any technical terms or jargon. If you must use a difficult word, explain it simply in parentheses.\n"
                "3. Provide exactly ONE clear, relatable real-life example (e.g., using toys, school, playground, or food) to illustrate the concept.\n"
                "4. Keep the explanation short and engaging.\n"
                "5. Do NOT use any introductory or concluding conversational filler."
            )),
            ("human", "Text to explain simply:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(text=text)
        
        try:
            explanation = self._invoke_llm(self.llm, formatted_prompt).strip()
            return explanation
        except Exception as e:
            return f"Error generating simple explanation: {str(e)}"

    def translate(self, text: str, target_language: str) -> str:
        """
        Translates the provided text into the target language using Llama 3.2.
        Maintains formatting, bullet points, and does not translate code blocks.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                f"You are a professional translator. Translate the provided text into {target_language}.\n"
                "Follow these strict rules:\n"
                "1. Maintain the exact same formatting, bullet points, headings, and paragraphs as the original text.\n"
                "2. Do NOT translate any code blocks (text inside triple backticks ```). Keep them exactly as they are in English.\n"
                "3. Do NOT translate technical terms that are commonly kept in English (e.g., HTML, API, PDF, RAG) if it makes the translation clearer.\n"
                "4. Output ONLY the translated text. Do NOT include any introductory or concluding translator notes."
            )),
            ("human", "Text to translate:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(text=text)
        
        try:
            translation = self._invoke_llm(self.llm, formatted_prompt).strip()
            return translation
        except Exception as e:
            return f"Error generating translation: {str(e)}"

    def get_page_content(self, page_num: int) -> Dict[str, Any]:
        """
        Retrieves all chunks from the vector store belonging to a specific page number.
        Returns a dictionary with page content and summary.
        """
        if self.vector_store is None:
            return {"error": "No documents uploaded yet."}
            
        page_chunks = []
        for doc in self.vector_store.docstore._dict.values():
            if doc.metadata.get("page") == page_num:
                page_chunks.append(doc)
                
        if not page_chunks:
            return {"error": f"Page {page_num} does not exist in the uploaded document(s)."}
            
        # Sort by chunk_index to maintain reading order
        page_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text
        extracted_text = "\n\n".join([doc.page_content for doc in page_chunks])
        
        # Generate summary of this specific page
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert assistant. Summarize the provided text from a specific page of a document.\n"
                "Provide a clear, concise summary of the main topics and key points discussed on this page.\n"
                "Keep the summary structured and easy to read. Do NOT include introductory or concluding conversational filler."
            )),
            ("human", "Page Text:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(text=extracted_text)
        
        try:
            summary = self._invoke_llm(self.llm, formatted_prompt).strip()
        except Exception as e:
            summary = f"Error generating page summary: {str(e)}"
            
        return {
            "page": page_num,
            "extracted_text": extracted_text,
            "summary": summary,
            "sources": [
                {
                    "source": os.path.basename(doc.metadata.get("source", "")),
                    "page": doc.metadata.get("page"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "snippet": doc.page_content[:200]
                } for doc in page_chunks
            ]
        }

    def compare_pdfs(self, file1: str, file2: str) -> str:
        """
        Retrieves all chunks for both PDFs, sorts them, and uses the local LLM
        with a LangChain template to generate a JSON comparison.
        """
        if self.vector_store is None:
            return "{}"
            
        # Retrieve chunks for file1 and file2
        chunks1 = []
        chunks2 = []
        
        for doc in self.vector_store.docstore._dict.values():
            source = doc.metadata.get("source", "")
            basename = os.path.basename(source)
            if basename == file1:
                chunks1.append(doc)
            elif basename == file2:
                chunks2.append(doc)
                
        if not chunks1 or not chunks2:
            return "{}"
            
        # Sort chunks
        chunks1.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        chunks2.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text (up to 18,000 characters each to prevent LLM context overflow)
        text1 = "\n\n".join([doc.page_content for doc in chunks1])
        if len(text1) > 18000:
            text1 = text1[:18000]
            
        text2 = "\n\n".join([doc.page_content for doc in chunks2])
        if len(text2) > 18000:
            text2 = text2[:18000]
            
        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert document analyst. Compare the two provided documents across these categories:\n"
                "1. Purpose\n"
                "2. Main Topics\n"
                "3. Important Concepts\n"
                "4. Differences\n"
                "5. Similarities\n"
                "6. Final Conclusion\n\n"
                "The output MUST be a valid JSON object. Do NOT include any markdown formatting (like ```json or ```), introductory text, or explanatory text. Output ONLY the raw JSON.\n\n"
                "The JSON object must have exactly the following keys and structure:\n"
                "{{\n"
                '  "purpose": {{\n'
                '    "doc1": "purpose of document 1",\n'
                '    "doc2": "purpose of document 2",\n'
                '    "comparison": "brief comparison of purposes"\n'
                "  }},\n"
                '  "topics": {{\n'
                '    "doc1": "main topics of document 1",\n'
                '    "doc2": "main topics of document 2",\n'
                '    "comparison": "brief comparison of topics"\n'
                "  }},\n"
                '  "concepts": {{\n'
                '    "doc1": "important concepts in document 1",\n'
                '    "doc2": "important concepts in document 2",\n'
                '    "comparison": "brief comparison of concepts"\n'
                "  }},\n"
                '  "differences": "Detailed list of differences",\n'
                '  "similarities": "Detailed list of similarities",\n'
                '  "conclusion": "Final comparison conclusion"\n'
                "}}\n"
            )),
            ("human", (
                "Document 1 Title: {file1}\n"
                "Document 1 Content:\n{text1}\n\n"
                "Document 2 Title: {file2}\n"
                "Document 2 Content:\n{text2}"
            ))
        ])
        
        formatted_prompt = prompt_template.format(
            file1=file1,
            text1=text1,
            file2=file2,
            text2=text2
        )
        
        try:
            comparison_output = self._invoke_llm(self.json_llm, formatted_prompt).strip()
            
            # Clean up markdown code blocks if the LLM included them
            if comparison_output.startswith("```"):
                lines = comparison_output.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                comparison_output = "\n".join(lines).strip()
                
            return comparison_output
        except Exception as e:
            print(f"Error comparing PDFs: {e}")
            return "{}"

    def generate_notes(self, filename: str) -> str:
        """
        Retrieves all chunks for a PDF, sorts them, and uses the local LLM
        with a LangChain template to generate a JSON containing five types of notes.
        """
        if self.vector_store is None:
            return "{}"
            
        file_chunks = []
        for doc in self.vector_store.docstore._dict.values():
            source = doc.metadata.get("source", "")
            if os.path.basename(source) == filename:
                file_chunks.append(doc)
                
        if not file_chunks:
            return "{}"
            
        # Sort chunks
        file_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text (up to 30,000 characters to prevent LLM context overflow)
        text = "\n\n".join([doc.page_content for doc in file_chunks])
        if len(text) > 30000:
            text = text[:30000]
            
        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert academic tutor and note-taker. Generate comprehensive notes from the provided text.\n"
                "You must generate exactly five types of notes:\n"
                "1. Short Notes: A concise, high-level summary of the document.\n"
                "2. Bullet Point Notes: A detailed breakdown of the main points, organized with clear bullet points.\n"
                "3. Important Concepts: A list of key terms, definitions, or core formulas/theories.\n"
                "4. Revision Notes: Quick-recall summaries, mnemonics, or flashcard-style Q&As for rapid review.\n"
                "5. Exam Preparation Notes: Potential exam questions (with answers) and key topics likely to be tested.\n\n"
                "The output MUST be a valid JSON object. Do NOT include any markdown formatting (like ```json or ```), introductory text, or explanatory text. Output ONLY the raw JSON.\n\n"
                "The JSON object must have exactly the following keys and structure:\n"
                "{{\n"
                '  "short_notes": "markdown formatted short notes",\n'
                '  "bullet_points": "markdown formatted bullet points",\n'
                '  "concepts": "markdown formatted important concepts",\n'
                '  "revision": "markdown formatted revision notes",\n'
                '  "exam_prep": "markdown formatted exam prep notes"\n'
                "}}\n"
            )),
            ("human", "Document Title: {filename}\nDocument Content:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(
            filename=filename,
            text=text
        )
        
        try:
            notes_output = self._invoke_llm(self.json_llm, formatted_prompt).strip()
            
            # Clean up markdown code blocks if the LLM included them
            if notes_output.startswith("```"):
                lines = notes_output.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                notes_output = "\n".join(lines).strip()
                
            return notes_output
        except Exception as e:
            print(f"Error generating notes: {e}")
            return "{}"

    def generate_interview_prep(self, filename: str) -> str:
        """
        Retrieves all chunks for a PDF, sorts them, and uses the local LLM
        with a LangChain template to generate a JSON containing 30 interview questions.
        """
        if self.vector_store is None:
            return "{}"
            
        file_chunks = []
        for doc in self.vector_store.docstore._dict.values():
            source = doc.metadata.get("source", "")
            if os.path.basename(source) == filename:
                file_chunks.append(doc)
                
        if not file_chunks:
            return "{}"
            
        # Sort chunks
        file_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text (up to 30,000 characters to prevent LLM context overflow)
        text = "\n\n".join([doc.page_content for doc in file_chunks])
        if len(text) > 30000:
            text = text[:30000]
            
        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert technical interviewer. Generate exactly 30 interview questions based on the provided text.\n"
                "The questions must be divided exactly as:\n"
                "- 10 Basic Questions (simple, foundational concepts)\n"
                "- 10 Intermediate Questions (application, analysis, practical scenarios)\n"
                "- 10 Advanced Questions (deep-dive, architecture, trade-offs, edge cases)\n\n"
                "For every question, provide:\n"
                "1. The question text\n"
                "2. A detailed expected answer\n"
                "3. The difficulty level (must be exactly 'Basic', 'Intermediate', or 'Advanced')\n"
                "4. A short topic tag (1-3 words)\n\n"
                "The output MUST be a valid JSON object. Do NOT include any markdown formatting (like ```json or ```), introductory text, or explanatory text. Output ONLY the raw JSON.\n\n"
                "The JSON object must have exactly the following keys and structure:\n"
                "{{\n"
                '  "questions": [\n'
                '    {{\n'
                '      "question": "question text",\n'
                '      "answer": "expected answer text",\n'
                '      "difficulty": "Basic",\n'
                '      "topic": "topic name"\n'
                "    }},\n"
                "    ...\n"
                "  ]\n"
                "}}\n"
            )),
            ("human", "Document Title: {filename}\nDocument Content:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(
            filename=filename,
            text=text
        )
        
        try:
            interview_output = self._invoke_llm(self.json_llm, formatted_prompt).strip()
            
            # Clean up markdown code blocks if the LLM included them
            if interview_output.startswith("```"):
                lines = interview_output.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                interview_output = "\n".join(lines).strip()
                
            return interview_output
        except Exception as e:
            print(f"Error generating interview prep: {e}")
            return "{}"

    def generate_flashcards(self, filename: str) -> str:
        """
        Retrieves all chunks for a PDF, sorts them, and uses the local LLM
        with a LangChain template to generate a JSON containing a list of flashcards.
        """
        if self.vector_store is None:
            return "{}"
            
        file_chunks = []
        for doc in self.vector_store.docstore._dict.values():
            source = doc.metadata.get("source", "")
            if os.path.basename(source) == filename:
                file_chunks.append(doc)
                
        if not file_chunks:
            return "{}"
            
        # Sort chunks
        file_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine text (up to 30,000 characters to prevent LLM context overflow)
        text = "\n\n".join([doc.page_content for doc in file_chunks])
        if len(text) > 30000:
            text = text[:30000]
            
        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert tutor. Generate exactly 10 high-quality study flashcards based on the provided text.\n"
                "Each flashcard must contain:\n"
                "1. Front: A clear, concise question or term.\n"
                "2. Back: A brief, accurate answer or definition.\n\n"
                "The output MUST be a valid JSON object. Do NOT include any markdown formatting (like ```json or ```), introductory text, or explanatory text. Output ONLY the raw JSON.\n\n"
                "The JSON object must have exactly the following keys and structure:\n"
                "{{\n"
                '  "flashcards": [\n'
                '    {{\n'
                '      "front": "Question or term on the front",\n'
                '      "back": "Answer or definition on the back"\n'
                '    }},\n'
                "    ...\n"
                "  ]\n"
                "}}\n"
            )),
            ("human", "Document Title: {filename}\nDocument Content:\n{text}")
        ])
        
        formatted_prompt = prompt_template.format(
            filename=filename,
            text=text
        )
        
        try:
            flashcard_output = self._invoke_llm(self.json_llm, formatted_prompt).strip()
            
            # Clean up markdown code blocks if the LLM included them
            if flashcard_output.startswith("```"):
                lines = flashcard_output.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                flashcard_output = "\n".join(lines).strip()
                
            return flashcard_output
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            return "{}"
