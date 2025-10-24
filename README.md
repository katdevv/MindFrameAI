# MindFrameAI RAG Engine

MindFrameAI is a personalized learning assistant that can read a file (PDF, DOCX, TXT, CSV, JSON), understand it, and generate:
- A direct answer to a user question
- Study context
- Georgian-language flashcards (in JSON)
- A Georgian multiple-choice quiz (in JSON)

It adapts tone, structure, and difficulty based on a learner profile.

This module is designed to be part of a larger AI study workspace with features like focus mode, flashcards, quizzes/tests, TTS, progress tracking, and file upload.

---

## ✨ Key Features

### 1. Retrieval-Augmented Generation (RAG)
- Loads a document and splits it into semantic chunks.
- Builds a FAISS vector store using OpenAI embeddings.
- Retrieves the most relevant chunks to answer a question.

### 2. Adaptive Pedagogy
- It builds a dynamic system prompt using the learner's self-description:
  - learning style (visual, hands-on, auditory, step-by-step)
  - pace preference (fast vs thorough)
  - level (beginner / intermediate / advanced)
- The assistant explains content in the learner’s preferred style.

### 3. Flashcards Generator (Georgian)
- Generates ~10 flashcards in pure JSON.
- The content of the flashcards (Q/A) is in Georgian.
- Automatically writes output to `flashcards.json`.

### 4. Quiz Generator (Georgian)
- Generates ~10 multiple-choice questions with options A/B/C/D, correct answer, and explanation.
- All question/answer text is in Georgian.
- Automatically writes output to `quiz.json`.

### 5. Async Pipeline with LangGraph
- The full workflow is modeled as a state machine:
  1. Load + chunk + embed file
  2. Retrieve relevant passages for a given query
  3. Generate answer
  4. Generate flashcards
  5. Generate quiz
- Each step updates a shared `RAGState` and passes it forward.

---
