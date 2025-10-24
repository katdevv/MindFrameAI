import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
import asyncio
import threading
from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import hashlib
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    Docx2txtLoader
)
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def make_system_prompt(QA: dict) -> str:
    learning_style_analysis = ""
    personality_traits = ""
    difficulty_level = "intermediate"

    for _, answer in QA.items():
        answer_lower = str(answer).lower()

        if "visual" in answer_lower or "diagram" in answer_lower or "chart" in answer_lower:
            learning_style_analysis += "Visual learner - prefers diagrams, charts, and visual representations. "
        if "hands-on" in answer_lower or "practice" in answer_lower or "doing" in answer_lower:
            learning_style_analysis += "Kinesthetic learner - learns best through hands-on practice. "
        if "explain" in answer_lower or "discuss" in answer_lower or "talk" in answer_lower:
            learning_style_analysis += "Auditory learner - benefits from explanations and discussions. "
        if "step" in answer_lower or "sequence" in answer_lower or "order" in answer_lower:
            learning_style_analysis += "Sequential learner - prefers step-by-step approaches. "

        if "beginner" in answer_lower or "basic" in answer_lower or "simple" in answer_lower:
            difficulty_level = "beginner"
        elif "advanced" in answer_lower or "complex" in answer_lower or "challenging" in answer_lower:
            difficulty_level = "advanced"

        if "quick" in answer_lower or "fast" in answer_lower:
            personality_traits += "Prefers fast-paced learning. "
        if "detail" in answer_lower or "thorough" in answer_lower:
            personality_traits += "Appreciates detailed explanations. "
        if "example" in answer_lower or "practical" in answer_lower:
            personality_traits += "Learns well with real-world examples. "

    system_prompt = f"""You are an AI-powered personalized learning assistant.

LEARNER PROFILE:
{learning_style_analysis if learning_style_analysis else "Mixed learning style - adapts to various approaches."}
{personality_traits if personality_traits else "Balanced learning pace with standard detail level."}
Difficulty Level: {difficulty_level}

PRINCIPLES:
1. Personalize to learner profile
2. Make it engaging
3. Progress step by step
4. Reinforce key points
5. Connect to real-world examples

COMMUNICATION:
- Encouraging, clear, concise
- Adjust complexity to learner level
- Use examples and visuals when possible
- Confirm understanding before moving on"""
    return system_prompt


class RAGState(TypedDict):
    file_path: str
    query: str
    context: str
    answer: str
    documents: List[Document]
    vector_store: Optional[Any]
    error: Optional[str]
    flashcards: Optional[str]
    quiz: Optional[str]
    user_qa: Optional[Dict]


class RAGNode:
    def __init__(
            self,
            openai_api_key: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            k: int = 4,
            model_name: str = "gpt-4"
    ):
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.model_name = model_name

        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.7
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self._vector_store_cache = {}

    def _get_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_documents(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext == ".json":
            loader = JSONLoader(file_path, jq_schema=".")
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    async def load_and_process_file(self, state: RAGState) -> RAGState:
        try:
            file_hash = self._get_file_hash(state["file_path"])
            if file_hash in self._vector_store_cache:
                state["vector_store"] = self._vector_store_cache[file_hash]
                state["documents"] = []
                return state

            documents = self._load_documents(state["file_path"])
            chunks = self.text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(chunks, self.embeddings)

            self._vector_store_cache[file_hash] = vector_store
            state["documents"] = chunks
            state["vector_store"] = vector_store
            state["error"] = None
            return state
        except Exception as e:
            state["error"] = f"Error in load_and_process_file: {str(e)}"
            return state

    async def retrieve_context(self, state: RAGState) -> RAGState:
        try:
            if state.get("error"):
                return state
            retriever = state["vector_store"].as_retriever(search_kwargs={"k": self.k})
            docs = retriever.invoke(state["query"])
            state["context"] = "\n\n".join([d.page_content for d in docs])
            return state
        except Exception as e:
            state["error"] = f"Error in retrieve_context: {str(e)}"
            return state

    async def generate_answer(self, state: RAGState) -> RAGState:
        try:
            if state.get("error"):
                return state
            system_prompt = make_system_prompt(state.get("user_qa", {}))
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nContext:\n{context}"),
                ("human", "{question}")
            ])
            chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
            )
            state["answer"] = await chain.ainvoke(
                {"context": state["context"], "question": state["query"]}
            )
            return state
        except Exception as e:
            state["error"] = f"Error in generate_answer: {str(e)}"
            return state

    async def generate_flash_cards(self, state: RAGState) -> RAGState:
        try:
            if state.get("error"):
                return state
            system_prompt = make_system_prompt(state.get("user_qa", {}))
            flashcard_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + """
Create 10 educational flashcards in JSON format based on the content.
Example format:
{{
  "flashcards": [
    {{"id": 1, "front": "What is artificial intelligence?", "back": "Artificial intelligence is ..."}}
  ]
}}
"""),
                ("human", "Create flashcards based on this content: {context}")
            ])
            chain = flashcard_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"context": state["context"]})
            state["flashcards"] = response
            return state
        except Exception as e:
            state["error"] = f"Error in generate_flash_cards: {str(e)}"
            return state

    async def generate_questions(self, state: RAGState) -> RAGState:
        try:
            if state.get("error"):
                return state
            system_prompt = make_system_prompt(state.get("user_qa", {}))
            quiz_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + """
Create a quiz of 10 multiple-choice questions in JSON format.
Example format:
{{
  "questions": [
    {{
      "id": 1,
      "question": "When was the first chatbot ELIZA created?",
      "options": {{"A": "1950", "B": "1960s", "C": "1977", "D": "2011"}},
      "correct_answer": "B",
      "explanation": "ELIZA was created in the 1960s as the first chatbot."
    }}
  ]
}}
"""),
                ("human", "Create quiz questions based on this content: {context}")
            ])
            chain = quiz_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"context": state["context"]})
            state["quiz"] = response
            return state
        except Exception as e:
            state["error"] = f"Error in generate_questions: {str(e)}"
            return state

    def create_rag_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        workflow.add_node("load_and_process", self.load_and_process_file)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("generate_flashcards", self.generate_flash_cards)
        workflow.add_node("generate_quiz", self.generate_questions)
        workflow.add_edge(START, "load_and_process")
        workflow.add_edge("load_and_process", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "generate_flashcards")
        workflow.add_edge("generate_flashcards", "generate_quiz")
        workflow.add_edge("generate_quiz", END)
        return workflow.compile()

    async def query_file(self, file_path: str, query: str, user_qa: Dict = None) -> Dict[str, Any]:
        init_state = RAGState(
            file_path=file_path,
            query=query,
            context="",
            answer="",
            documents=[],
            vector_store=None,
            error=None,
            flashcards=None,
            quiz=None,
            user_qa=user_qa or {}
        )
        app = self.create_rag_graph()
        result = await app.ainvoke(init_state)
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", ""),
            "flashcards": result.get("flashcards", ""),
            "quiz": result.get("quiz", ""),
            "error": result.get("error"),
            "file_path": file_path,
            "query": query
        }


class AILearningAssistant:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Learning Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Application state
        self.user_answers = {}
        self.selected_file_path = None
        self.rag_node = None
        self.questions_data = self.load_questions()

        # Initialize RAG node
        self.initialize_rag_node()

        # Start with questionnaire screen
        self.show_questionnaire()

    def initialize_rag_node(self):
        """Initialize the RAG node with OpenAI API key"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                messagebox.showerror("Error",
                                     "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return
            self.rag_node = RAGNode(openai_api_key=api_key)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize AI model: {str(e)}")

    def load_questions(self):
        """Load questions from the embedded JSON data"""
        return {
            "questions": [
                {
                    "id": 1,
                    "question": "How do you usually start preparing for a new topic?",
                    "answers": [
                        "Reading through the material",
                        "Watching videos/lectures",
                        "Making notes or highlights",
                        "Jumping straight into practice questions"
                    ]
                },
                {
                    "id": 2,
                    "question": "Which study method do you find most effective?",
                    "answers": [
                        "Taking notes",
                        "Making flashcards",
                        "Summarizing or rewriting content",
                        "Group discussions/study with peers"
                    ]
                },
                {
                    "id": 3,
                    "question": "How long can you usually stay focused before getting distracted?",
                    "answers": [
                        "Less than 15 minutes",
                        "15–30 minutes",
                        "30–60 minutes",
                        "More than an hour"
                    ]
                },
                {
                    "id": 4,
                    "question": "How do you prefer to test your knowledge?",
                    "answers": [
                        "Practice quizzes",
                        "Flashcards",
                        "Explaining concepts to someone else",
                        "Writing summaries"
                    ]
                },
                {
                    "id": 5,
                    "question": "What distracts you the most while studying?",
                    "answers": [
                        "Phone/social media",
                        "Noise or environment",
                        "Lack of motivation",
                        "Not knowing where to start"
                    ]
                },
                {
                    "id": 6,
                    "question": "How do you usually keep track of your study progress?",
                    "answers": [
                        "To-do lists",
                        "Timetables or planners",
                        "Apps or digital tools",
                        "I don't track progress"
                    ]
                },
                {
                    "id": 7,
                    "question": "When do you usually study best?",
                    "answers": [
                        "Early morning",
                        "Afternoon",
                        "Evening",
                        "Late at night"
                    ]
                },
                {
                    "id": 8,
                    "question": "What motivates you most to study?",
                    "answers": [
                        "Upcoming exams/tests",
                        "Long-term goals (career, knowledge)",
                        "External pressure (family, teachers)",
                        "Rewards or breaks (music, games, snacks)"
                    ]
                },
                {
                    "id": 9,
                    "question": "Which study environment helps you focus best?",
                    "answers": [
                        "Quiet room alone",
                        "Library or study space",
                        "With background music/noise",
                        "With friends or study group"
                    ]
                },
                {
                    "id": 10,
                    "question": "If you could improve one part of your study routine, what would it be?",
                    "answers": [
                        "Staying focused",
                        "Having a clear study plan",
                        "Tracking progress",
                        "Finding engaging ways to learn"
                    ]
                }
            ]
        }

    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_questionnaire(self):
        """Display the questionnaire screen"""
        self.clear_window()

        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(main_frame, text="🎓 Learning Style Assessment",
                               font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))

        # Subtitle
        subtitle_label = tk.Label(main_frame,
                                  text="Help us understand your learning preferences to personalize your experience",
                                  font=('Arial', 12), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 30))

        # Scrollable frame for questions
        canvas = tk.Canvas(main_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create questions
        self.question_vars = {}
        for i, question in enumerate(self.questions_data["questions"]):
            question_frame = ttk.LabelFrame(scrollable_frame, text=f"Question {i + 1}", padding="15")
            question_frame.pack(fill=tk.X, pady=10, padx=10)

            # Question text
            q_label = tk.Label(question_frame, text=question["question"],
                               font=('Arial', 11, 'bold'), bg='#f0f0f0', wraplength=600, justify=tk.LEFT)
            q_label.pack(anchor=tk.W, pady=(0, 10))

            # Answer options
            var = tk.StringVar()
            self.question_vars[question["id"]] = var

            for answer in question["answers"]:
                rb = ttk.Radiobutton(question_frame, text=answer, variable=var, value=answer)
                rb.pack(anchor=tk.W, pady=2)

        canvas.pack(side="left", fill="both", expand=True, padx=(0, 10))
        scrollbar.pack(side="right", fill="y")

        # Submit button
        submit_frame = ttk.Frame(main_frame)
        submit_frame.pack(fill=tk.X, pady=20)

        submit_btn = tk.Button(submit_frame, text="Submit & Continue",
                               command=self.submit_questionnaire,
                               font=('Arial', 14, 'bold'), bg='#3498db', fg='white',
                               relief=tk.FLAT, cursor='hand2')
        submit_btn.pack()

    def submit_questionnaire(self):
        """Process questionnaire submission"""
        # Collect answers
        for q_id, var in self.question_vars.items():
            if var.get():
                question = next(q for q in self.questions_data["questions"] if q["id"] == q_id)
                self.user_answers[question["question"]] = var.get()

        # Check if all questions are answered
        if len(self.user_answers) < len(self.questions_data["questions"]):
            messagebox.showwarning("Incomplete", "Please answer all questions before continuing.")
            return

        # Show file upload screen
        self.show_file_upload()

    def show_file_upload(self):
        """Display the file upload and query screen"""
        self.clear_window()

        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(main_frame, text="📁 Upload Learning Material",
                               font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))

        # File upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Select File", padding="20")
        upload_frame.pack(fill=tk.X, pady=(0, 20))

        self.file_label = tk.Label(upload_frame, text="No file selected",
                                   font=('Arial', 11), bg='#f0f0f0', fg='#7f8c8d')
        self.file_label.pack(pady=(0, 10))

        file_btn = tk.Button(upload_frame, text="📂 Browse Files",
                             command=self.browse_file,
                             font=('Arial', 12), bg='#2ecc71', fg='white',
                             relief=tk.FLAT, cursor='hand2')
        file_btn.pack()

        # Query section
        query_frame = ttk.LabelFrame(main_frame, text="Ask a Question", padding="20")
        query_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        query_label = tk.Label(query_frame, text="What would you like to know about this material?",
                               font=('Arial', 11), bg='#f0f0f0')
        query_label.pack(anchor=tk.W, pady=(0, 10))

        self.query_text = scrolledtext.ScrolledText(query_frame, height=5, font=('Arial', 11))
        self.query_text.pack(fill=tk.BOTH, expand=True)

        # Process button
        process_btn = tk.Button(main_frame, text="🚀 Process & Analyze",
                                command=self.process_file,
                                font=('Arial', 14, 'bold'), bg='#e74c3c', fg='white',
                                relief=tk.FLAT, cursor='hand2')
        process_btn.pack(pady=10)

    def browse_file(self):
        """Browse and select a file"""
        file_types = [
            ("All supported", "*.pdf;*.txt;*.docx;*.doc;*.csv;*.json"),
            ("PDF files", "*.pdf"),
            ("Text files", "*.txt"),
            ("Word documents", "*.docx;*.doc"),
            ("CSV files", "*.csv"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select a file to analyze",
            filetypes=file_types
        )

        if file_path:
            self.selected_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {filename}", fg='#27ae60')

    def process_file(self):
        """Process the uploaded file with the user's query"""
        if not self.selected_file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return

        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showerror("Error", "Please enter a question.")
            return

        if not self.rag_node:
            messagebox.showerror("Error", "AI model not initialized. Check your OpenAI API key.")
            return

        # Show processing screen
        self.show_processing_screen()

        # Process in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.run_rag_processing, args=(query,))
        thread.daemon = True
        thread.start()

    def show_processing_screen(self):
        """Show processing/loading screen"""
        self.clear_window()

        # Main frame
        main_frame = ttk.Frame(self.root, padding="50")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Loading animation
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start()

        # Loading text
        loading_label = tk.Label(main_frame, text="🤖 AI is analyzing your content...",
                                 font=('Arial', 16), bg='#f0f0f0', fg='#2c3e50')
        loading_label.pack(pady=20)

        status_label = tk.Label(main_frame, text="This may take a few moments",
                                font=('Arial', 12), bg='#f0f0f0', fg='#7f8c8d')
        status_label.pack()

    def run_rag_processing(self, query):
        """Run RAG processing in asyncio loop"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async query
            result = loop.run_until_complete(
                self.rag_node.query_file(self.selected_file_path, query, self.user_answers)
            )

            # Update UI in main thread
            self.root.after(0, lambda: self.show_results(result))

        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            self.root.after(0, lambda: self.show_error(error_msg))
        finally:
            loop.close()

    def show_error(self, error_msg):
        """Show error message"""
        self.clear_window()

        main_frame = ttk.Frame(self.root, padding="50")
        main_frame.pack(fill=tk.BOTH, expand=True)

        error_label = tk.Label(main_frame, text="❌ Error",
                               font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#e74c3c')
        error_label.pack(pady=20)

        error_text = tk.Label(main_frame, text=error_msg,
                              font=('Arial', 12), bg='#f0f0f0', fg='#7f8c8d',
                              wraplength=600, justify=tk.CENTER)
        error_text.pack(pady=20)

        retry_btn = tk.Button(main_frame, text="Try Again",
                              command=self.show_file_upload,
                              font=('Arial', 12), bg='#3498db', fg='white',
                              relief=tk.FLAT, cursor='hand2')
        retry_btn.pack()

    def show_results(self, result):
        """Display the analysis results"""
        self.clear_window()

        # Main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Answer tab
        answer_frame = ttk.Frame(notebook, padding="20")
        notebook.add(answer_frame, text="📝 Answer")

        answer_title = tk.Label(answer_frame, text="AI Analysis Result",
                                font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        answer_title.pack(anchor=tk.W, pady=(0, 15))

        answer_text = scrolledtext.ScrolledText(answer_frame, font=('Arial', 11),
                                                wrap=tk.WORD, state=tk.DISABLED)
        answer_text.pack(fill=tk.BOTH, expand=True)

        # Insert answer
        answer_text.config(state=tk.NORMAL)
        answer_text.insert(tk.END, result.get("answer", "No answer generated"))
        answer_text.config(state=tk.DISABLED)

        # Flashcards tab
        flashcards_frame = ttk.Frame(notebook, padding="20")
        notebook.add(flashcards_frame, text="🗃️ Flashcards")

        flashcards_text = scrolledtext.ScrolledText(flashcards_frame, font=('Arial', 11),
                                                    wrap=tk.WORD, state=tk.DISABLED)
        flashcards_text.pack(fill=tk.BOTH, expand=True)

        flashcards_text.config(state=tk.NORMAL)
        flashcards_text.insert(tk.END, result.get("flashcards", "No flashcards generated"))
        flashcards_text.config(state=tk.DISABLED)

        # Quiz tab
        quiz_frame = ttk.Frame(notebook, padding="20")
        notebook.add(quiz_frame, text="❓ Quiz")

        quiz_text = scrolledtext.ScrolledText(quiz_frame, font=('Arial', 11),
                                              wrap=tk.WORD, state=tk.DISABLED)
        quiz_text.pack(fill=tk.BOTH, expand=True)

        quiz_text.config(state=tk.NORMAL)
        quiz_text.insert(tk.END, result.get("quiz", "No quiz generated"))
        quiz_text.config(state=tk.DISABLED)

        # Info tab
        info_frame = ttk.Frame(notebook, padding="20")
        notebook.add(info_frame, text="ℹ️ Info")

        info_text = tk.Text(info_frame, font=('Arial', 11), wrap=tk.WORD, state=tk.DISABLED)
        info_text.pack(fill=tk.BOTH, expand=True)

        # Display file path and query info
        info_content = f"""File Path: {result.get('file_path', 'N/A')}

Query: {result.get('query', 'N/A')}

Learning Profile Summary:
{self.get_learning_profile_summary()}

Processing Status: {'Success' if not result.get('error') else 'Error'}

{f"Error Details: {result.get('error')}" if result.get('error') else ""}
"""

        info_text.config(state=tk.NORMAL)
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)

        # Bottom buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))

        # New Query button
        new_query_btn = tk.Button(buttons_frame, text="🔄 New Query",
                                  command=self.show_file_upload,
                                  font=('Arial', 12), bg='#3498db', fg='white',
                                  relief=tk.FLAT, cursor='hand2')
        new_query_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Save Results button
        save_btn = tk.Button(buttons_frame, text="💾 Save Results",
                             command=lambda: self.save_results(result),
                             font=('Arial', 12), bg='#27ae60', fg='white',
                             relief=tk.FLAT, cursor='hand2')
        save_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Restart button
        restart_btn = tk.Button(buttons_frame, text="🔄 New Assessment",
                                command=self.restart_application,
                                font=('Arial', 12), bg='#f39c12', fg='white',
                                relief=tk.FLAT, cursor='hand2')
        restart_btn.pack(side=tk.LEFT)

    def get_learning_profile_summary(self):
        """Generate a summary of the user's learning profile"""
        if not self.user_answers:
            return "No learning profile data available."

        summary = "Based on your responses:\n"

        # Analyze answers for key insights
        answers_text = " ".join(self.user_answers.values()).lower()

        insights = []

        if "visual" in answers_text or "diagram" in answers_text:
            insights.append("• Visual learning preference detected")
        if "practice" in answers_text or "hands-on" in answers_text:
            insights.append("• Kinesthetic/practical learning style")
        if "notes" in answers_text or "writing" in answers_text:
            insights.append("• Note-taking and writing-based learning")
        if "group" in answers_text or "discussion" in answers_text:
            insights.append("• Collaborative learning preference")
        if "quick" in answers_text or "fast" in answers_text:
            insights.append("• Prefers fast-paced learning")
        if "detail" in answers_text or "thorough" in answers_text:
            insights.append("• Appreciates detailed explanations")

        if insights:
            summary += "\n".join(insights)
        else:
            summary += "• Balanced learning approach with mixed preferences"

        return summary

    def save_results(self, result):
        """Save the results to files"""
        try:
            # Create results directory if it doesn't exist
            results_dir = "ai_learning_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Save main answer
            with open(os.path.join(results_dir, "answer.txt"), "w", encoding="utf-8") as f:
                f.write("AI Learning Assistant Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"File: {result.get('file_path', 'N/A')}\n")
                f.write(f"Query: {result.get('query', 'N/A')}\n\n")
                f.write("ANSWER:\n")
                f.write("-" * 20 + "\n")
                f.write(result.get("answer", "No answer generated"))

            # Save flashcards
            flashcards_content = result.get("flashcards", "")
            if flashcards_content:
                try:
                    # Try to parse as JSON for better formatting
                    flashcards_json = json.loads(flashcards_content)
                    with open(os.path.join(results_dir, "flashcards.json"), "w", encoding="utf-8") as f:
                        json.dump(flashcards_json, f, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    # Save as plain text if not valid JSON
                    with open(os.path.join(results_dir, "flashcards.txt"), "w", encoding="utf-8") as f:
                        f.write(flashcards_content)

            # Save quiz
            quiz_content = result.get("quiz", "")
            if quiz_content:
                try:
                    # Try to parse as JSON for better formatting
                    quiz_json = json.loads(quiz_content)
                    with open(os.path.join(results_dir, "quiz.json"), "w", encoding="utf-8") as f:
                        json.dump(quiz_json, f, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    # Save as plain text if not valid JSON
                    with open(os.path.join(results_dir, "quiz.txt"), "w", encoding="utf-8") as f:
                        f.write(quiz_content)

            # Save learning profile
            with open(os.path.join(results_dir, "learning_profile.txt"), "w", encoding="utf-8") as f:
                f.write("Learning Profile Assessment\n")
                f.write("=" * 30 + "\n\n")
                for question, answer in self.user_answers.items():
                    f.write(f"Q: {question}\n")
                    f.write(f"A: {answer}\n\n")
                f.write("\nProfile Summary:\n")
                f.write("-" * 15 + "\n")
                f.write(self.get_learning_profile_summary())

            messagebox.showinfo("Success", f"Results saved successfully to '{results_dir}' folder!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def restart_application(self):
        """Restart the application from the beginning"""
        self.user_answers = {}
        self.selected_file_path = None
        self.question_vars = {}
        self.show_questionnaire()

    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the AI Learning Assistant?"):
            self.root.destroy()


if __name__ == "__main__":
    # Check if required dependencies are available
    try:
        import langchain_openai
        import langgraph
        import langchain_community
    except ImportError as e:
        print("Error: Missing required dependencies.")
        print("Please install the required packages:")
        print("pip install langchain langchain-openai langchain-community langgraph faiss-cpu python-dotenv")
        print(f"Missing module: {e.name}")
        exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not found.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("Or create a .env file with: OPENAI_API_KEY=your-api-key-here")

        # Allow user to continue anyway (they might set it in a .env file)
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(1)

    # Start the application
    app = AILearningAssistant()
    app.run()