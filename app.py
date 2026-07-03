import os
import shutil
import json
import uuid
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

from chatbot.pdf_loader import load_and_split_pdf
from chatbot.rag_pipeline import RAGPipeline

# Initialize Flask App
app = Flask(__name__)

# Configurations
if os.environ.get("VERCEL"):
    UPLOAD_FOLDER = '/tmp/uploads'
    CHAT_HISTORY_FILE = '/tmp/chat_history.json'
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    CHAT_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_history.json')

ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize RAG Pipeline
# We'll use "llama3.2" as the default model.
rag_pipeline = RAGPipeline(model_name="llama3.2")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_uploaded_files():
    """
    Returns a list of files in the uploads folder.
    """
    if not os.path.exists(UPLOAD_FOLDER):
        return []
    return [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]

# Chat History Helper Functions
def load_chats():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return {}
    try:
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_chats(chats):
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chats, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chats: {e}")

@app.route('/')
def index():
    uploaded_files = get_uploaded_files()
    has_index = rag_pipeline.vector_store is not None
    return render_template('index.html', uploaded_files=uploaded_files, has_index=has_index)

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    """
    Serves the uploaded PDF files so they can be viewed in the UI iframe.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        try:
            # Process the PDF: Load and Split
            chunks = load_and_split_pdf(file_path)
            
            if not chunks:
                return jsonify({"error": "The PDF file appears to be empty or has no extractable text."}), 400
                
            # Add to the RAG Pipeline vector store
            rag_pipeline.add_documents(chunks)
            
            return jsonify({
                "message": f"Successfully processed '{filename}' ({len(chunks)} text chunks indexed).",
                "filename": filename,
                "uploaded_files": get_uploaded_files()
            })
            
        except Exception as e:
            # If processing fails, remove the file to keep state clean
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
            
    return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """
    Deletes a specific PDF file and rebuilds the FAISS index from remaining files.
    """
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400
        
    filename = secure_filename(data['filename'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        # Delete file
        os.remove(file_path)
        
        # Clear and rebuild database
        rag_pipeline.clear_database()
        
        remaining_files = get_uploaded_files()
        total_chunks = 0
        for f in remaining_files:
            path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            chunks = load_and_split_pdf(path)
            if chunks:
                rag_pipeline.add_documents(chunks)
                total_chunks += len(chunks)
                
        return jsonify({
            "message": f"Successfully deleted '{filename}' and rebuilt index.",
            "uploaded_files": remaining_files,
            "has_index": len(remaining_files) > 0
        })
    except Exception as e:
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Standard non-streaming chat route (retained for compatibility).
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
        
    user_message = data['message']
    chat_history = data.get('chat_history', [])
    
    try:
        response = rag_pipeline.answer_question(user_message, chat_history)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streams the response token-by-token and yields status updates.
    """
    data = request.get_json()
    if not data or 'message' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing message or chat_id"}), 400
        
    user_message = data['message']
    chat_id = data['chat_id']
    
    chats = load_chats()
    if chat_id not in chats:
        return jsonify({"error": "Chat session not found"}), 404
        
    chat_history = chats[chat_id].get("messages", [])
    
    def generate():
        full_answer = ""
        sources = []
        
        for event in rag_pipeline.answer_question_stream(user_message, chat_history):
            if event["type"] == "status":
                yield f"event: status\ndata: {json.dumps(event['content'])}\n\n"
            elif event["type"] == "token":
                full_answer += event["content"]
                yield f"event: token\ndata: {json.dumps(event['content'])}\n\n"
            elif event["type"] == "sources":
                sources = event["content"]
                yield f"event: sources\ndata: {json.dumps(event['content'])}\n\n"
                
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            if chats[chat_id]["title"] == "New Chat" and len(chats[chat_id]["messages"]) == 0:
                title = user_message[:25] + "..." if len(user_message) > 25 else user_message
                chats[chat_id]["title"] = title
                
            chats[chat_id]["messages"].append({"role": "user", "text": user_message})
            chats[chat_id]["messages"].append({"role": "ai", "text": full_answer, "sources": sources})
            save_chats(chats)
            
        yield f"event: done\ndata: {json.dumps({'title': chats[chat_id]['title'] if chat_id in chats else 'New Chat'})}\n\n"
        
    return Response(generate(), mimetype='text/event-stream')

@app.route('/summarize', methods=['POST'])
def summarize_document():
    """
    Generates a structured summary for a specific PDF.
    """
    data = request.get_json()
    if not data or 'filename' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing filename or chat_id"}), 400
        
    filename = secure_filename(data['filename'])
    chat_id = data['chat_id']
    
    try:
        summary = rag_pipeline.summarize_pdf(filename)
        
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            chats[chat_id]["messages"].append({
                "role": "ai", 
                "text": f"### Document Summary: {filename}\n\n{summary}"
            })
            save_chats(chats)
            
        return jsonify({
            "summary": summary,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """
    Generates a 10-question multiple-choice quiz for a specific PDF.
    """
    data = request.get_json()
    if not data or 'filename' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing filename or chat_id"}), 400
        
    filename = secure_filename(data['filename'])
    chat_id = data['chat_id']
    
    try:
        raw_quiz = rag_pipeline.generate_quiz(filename)
        
        # Parse it to make sure it's valid JSON
        quiz_data = json.loads(raw_quiz)
        
        # Save to chat history as a special message block
        chats = load_chats()
        if chat_id in chats:
            chats[chat_id]["messages"].append({
                "role": "ai",
                "text": f"### Quiz Generated: {filename}",
                "quiz": quiz_data  # Store the raw quiz data
            })
            save_chats(chats)
            
        return jsonify({
            "quiz": quiz_data,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate quiz: {str(e)}"}), 500

@app.route('/explain_simply', methods=['POST'])
def explain_simply_route():
    """
    Simplifies complex text into a 10-year-old child's level.
    """
    data = request.get_json()
    if not data or 'text' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing text or chat_id"}), 400
        
    text = data['text']
    chat_id = data['chat_id']
    message_index = data.get('message_index', None)
    save_as_new = data.get('save_as_new', False)
    
    try:
        explanation = rag_pipeline.explain_simply(text)
        
        # Save to chat history
        if message_index is not None:
            chats = load_chats()
            if chat_id in chats and 0 <= message_index < len(chats[chat_id]["messages"]):
                chats[chat_id]["messages"][message_index]["eli10"] = explanation
                save_chats(chats)
        elif save_as_new:
            chats = load_chats()
            if chat_id in chats:
                chats[chat_id]["messages"].append({
                    "role": "ai",
                    "text": f"### Simple Explanation:\n\n{explanation}"
                })
                save_chats(chats)
                
        return jsonify({
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": f"Failed to simplify text: {str(e)}"}), 500

@app.route('/translate', methods=['POST'])
def translate_route():
    """
    Translates AI-generated answers into the target language.
    """
    data = request.get_json()
    if not data or 'text' not in data or 'target_language' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing text, target_language, or chat_id"}), 400
        
    text = data['text']
    target_language = data['target_language']
    chat_id = data['chat_id']
    message_index = data.get('message_index', None)
    
    try:
        translation = rag_pipeline.translate(text, target_language)
        
        # Save to chat history if message_index is provided
        if message_index is not None:
            chats = load_chats()
            if chat_id in chats and 0 <= message_index < len(chats[chat_id]["messages"]):
                msg = chats[chat_id]["messages"][message_index]
                if "translations" not in msg:
                    msg["translations"] = {}
                msg["translations"][target_language] = translation
                save_chats(chats)
                
        return jsonify({
            "translation": translation,
            "target_language": target_language
        })
    except Exception as e:
        return jsonify({"error": f"Failed to translate text: {str(e)}"}), 500

@app.route('/page_search', methods=['POST'])
def page_search_route():
    """
    Retrieves and summarizes content from a specific page, and optionally answers a question.
    """
    data = request.get_json()
    if not data or 'page_num' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing page_num or chat_id"}), 400
        
    page_num = int(data['page_num'])
    chat_id = data['chat_id']
    question = data.get('question', '').strip()
    
    try:
        result = rag_pipeline.get_page_content(page_num)
        if "error" in result:
            return jsonify({"error": result["error"]}), 404
            
        extracted_text = result["extracted_text"]
        summary = result["summary"]
        
        # Answer the question using only the page text if a specific question is asked
        answer = None
        is_generic = any(phrase in question.lower() for phrase in ["what is written", "what is on", "summarize this page", "explain this page", "what's on this page"])
        if question and not is_generic:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an AI assistant. Answer the user's question using ONLY the provided page text.\n"
                    "If the answer cannot be found in the text, reply: 'I couldn't find this information on the selected page.'\n"
                    "Do not make up information."
                )),
                ("human", "Page Text:\n{context}\n\nQuestion: {question}")
            ])
            formatted_prompt = prompt_template.format(context=extracted_text, question=question)
            try:
                answer = rag_pipeline.llm.invoke(formatted_prompt).strip()
            except Exception as e:
                answer = f"Error answering question: {str(e)}"
                
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            page_search_data = {
                "page": page_num,
                "extracted_text": extracted_text,
                "summary": summary,
                "question": question if answer else None,
                "answer": answer
            }
            chats[chat_id]["messages"].append({
                "role": "ai",
                "text": f"### Page Search: Page {page_num}",
                "page_search": page_search_data
            })
            save_chats(chats)
            
        return jsonify({
            "page": page_num,
            "page_search": page_search_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to perform page search: {str(e)}"}), 500

@app.route('/compare_pdfs', methods=['POST'])
def compare_documents():
    """
    Compares two PDF documents.
    """
    data = request.get_json()
    if not data or 'file1' not in data or 'file2' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing file1, file2, or chat_id"}), 400
        
    file1 = secure_filename(data['file1'])
    file2 = secure_filename(data['file2'])
    chat_id = data['chat_id']
    
    try:
        raw_comparison = rag_pipeline.compare_pdfs(file1, file2)
        
        # Parse it to make sure it's valid JSON
        comparison_data = json.loads(raw_comparison)
        
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            chats[chat_id]["messages"].append({
                "role": "ai",
                "text": f"### Document Comparison: {file1} vs {file2}",
                "comparison": comparison_data
            })
            save_chats(chats)
            
        return jsonify({
            "comparison": comparison_data,
            "file1": file1,
            "file2": file2
        })
    except Exception as e:
        return jsonify({"error": f"Failed to compare documents: {str(e)}"}), 500

@app.route('/generate_notes', methods=['POST'])
def generate_notes_route():
    """
    Generates study notes from a specific PDF document.
    """
    data = request.get_json()
    if not data or 'filename' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing filename or chat_id"}), 400
        
    filename = secure_filename(data['filename'])
    chat_id = data['chat_id']
    
    try:
        raw_notes = rag_pipeline.generate_notes(filename)
        notes_data = json.loads(raw_notes)
        
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            chats[chat_id]["messages"].append({
                "role": "ai",
                "text": f"### Notes Generated: {filename}",
                "notes": notes_data
            })
            save_chats(chats)
            
        return jsonify({
            "notes": notes_data,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate study notes: {str(e)}"}), 500

@app.route('/generate_interview', methods=['POST'])
def generate_interview_route():
    """
    Generates interview questions and answers from a specific PDF document.
    """
    data = request.get_json()
    if not data or 'filename' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing filename or chat_id"}), 400
        
    filename = secure_filename(data['filename'])
    chat_id = data['chat_id']
    
    try:
        raw_interview = rag_pipeline.generate_interview_prep(filename)
        interview_data = json.loads(raw_interview)
        
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            chats[chat_id]["messages"].append({
                "role": "ai",
                "text": f"### Interview Prep Generated: {filename}",
                "interview_prep": interview_data
            })
            save_chats(chats)
            
        return jsonify({
            "interview_prep": interview_data,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate interview preparation: {str(e)}"}), 500

@app.route('/generate_flashcards', methods=['POST'])
def generate_flashcards_route():
    """
    Generates flashcards from a specific PDF document.
    """
    data = request.get_json()
    if not data or 'filename' not in data or 'chat_id' not in data:
        return jsonify({"error": "Missing filename or chat_id"}), 400
        
    filename = secure_filename(data['filename'])
    chat_id = data['chat_id']
    
    try:
        raw_flashcards = rag_pipeline.generate_flashcards(filename)
        flashcards_data = json.loads(raw_flashcards)
        
        # Save to chat history
        chats = load_chats()
        if chat_id in chats:
            chats[chat_id]["messages"].append({
                "role": "ai",
                "text": f"### Flashcards Generated: {filename}",
                "flashcards": flashcards_data
            })
            save_chats(chats)
            
        return jsonify({
            "flashcards": flashcards_data,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate flashcards: {str(e)}"}), 500

# Chat History Management Routes
@app.route('/chats', methods=['GET'])
def get_chats():
    return jsonify(load_chats())

@app.route('/chats', methods=['POST'])
def create_chat():
    chats = load_chats()
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {
        "id": chat_id,
        "title": "New Chat",
        "messages": []
    }
    save_chats(chats)
    return jsonify(chats[chat_id])

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found"}), 404

@app.route('/clear', methods=['POST'])
def clear_data():
    try:
        # Clear vector store
        rag_pipeline.clear_database()
        
        # Clear uploads folder
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
        # Clear chat history
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
            
        return jsonify({
            "message": "All uploaded documents, index, and chat history cleared successfully.",
            "uploaded_files": []
        })
    except Exception as e:
        return jsonify({"error": f"Failed to clear data: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def check_status():
    """
    Checks if the local Ollama service is running and if the model is available.
    """
    import urllib.request
    
    ollama_url = "http://localhost:11434/api/tags"
    status = {
        "ollama_running": False,
        "model_available": False,
        "installed_models": []
    }
    
    try:
        with urllib.request.urlopen(ollama_url, timeout=2) as response:
            if response.status == 200:
                status["ollama_running"] = True
                data = json.loads(response.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                status["installed_models"] = models
                
                target = rag_pipeline.model_name
                status["model_available"] = any(target in m for m in models)
    except Exception:
        pass
        
    return jsonify(status)

if __name__ == '__main__':
    print("Starting AI PDF Chatbot Server...")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    app.run(debug=True, port=5000)
