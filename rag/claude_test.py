import json
import os
import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import hashlib
from prompts import prompts
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
Create 10 educational flashcards in JSON:
{{
  "flashcards": [
    {{"id": 1, "front": "Question", "back": "Answer"}}
  ]
}}
"""),
                ("human", "Create flashcards based on this content: {context}")
            ])
            chain = flashcard_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"context": state["context"]})
            state["flashcards"] = response

            try:
                data = json.loads(response)
                with open("flashcards.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            except json.JSONDecodeError:
                with open("flashcards.json", "w", encoding="utf-8") as f:
                    json.dump({"flashcards": [], "raw_response": response}, f, indent=4)
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
Create a quiz of 10 MCQs in JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Question",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation": "Reason"
    }}
  ]
}}
"""),
                ("human", "Create quiz questions based on this content: {context}")
            ])
            chain = quiz_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"context": state["context"]})
            state["quiz"] = response

            try:
                data = json.loads(response)
                with open("quiz.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            except json.JSONDecodeError:
                with open("quiz.json", "w", encoding="utf-8") as f:
                    json.dump({"questions": [], "raw_response": response}, f, indent=4)
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


if __name__ == "__main__":
    async def main():
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-key-here"
        user_qa = {
            "What's your preferred learning style?": "I like visual diagrams and step-by-step explanations",
            "How do you like information presented?": "I prefer detailed examples with practical applications",
            "What's your current level with this topic?": "I'm a beginner but I learn quickly",
            "Do you prefer quick overviews or detailed explanations?": "I like thorough, detailed explanations",
            "How do you best remember information?": "Through hands-on practice and real examples"
        }

        rag_node = RAGNode(openai_api_key=OPENAI_API_KEY)
        result = await rag_node.query_file(
            file_path="/home/blux/Desktop/LocalAssistantProject/agent/A_Brief_Introduction_To_AI.pdf",
            query="Explain the main concepts in this document",
            user_qa=user_qa
        )

        print(f"Answer: {result['answer']}")
        print(f"Flashcards saved: {'flashcards.json' if result['flashcards'] else 'None'}")
        print(f"Quiz saved: {'quiz.json' if result['quiz'] else 'None'}")
        print(f"Error: {result['error']}")

    asyncio.run(main())