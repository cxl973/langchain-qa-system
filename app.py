"""
A context-aware Q&A system built with LangChain and LangGraph.
"""
import os
from datetime import datetime
from typing import TypedDict, List, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============ State Definition ============
class QAState(TypedDict):
    """State for the QA system with conversation history."""
    messages: List[BaseMessage]
    question: str
    context: str
    answer: str
    needs_context: bool


# ============ Nodes ============
def analyze_question(state: QAState) -> QAState:
    """Analyze the question to determine if context is needed."""
    question = state["question"]
    
    # Simple heuristic: check if question contains pronouns or references
    context_indicators = ["它", "这个", "那个", "之前", "刚才", "上面", "what", "this", "that", "it", "they"]
    needs_context = any(indicator in question.lower() for indicator in context_indicators)
    
    return {"needs_context": needs_context}


def retrieve_context(state: QAState) -> QAState:
    """Retrieve relevant context from conversation history."""
    messages = state["messages"]
    
    # Get last few messages as context (excluding current question)
    recent_messages = messages[:-1] if messages else []
    context_parts = []
    
    for msg in recent_messages[-5:]:  # Last 5 messages
        if isinstance(msg, HumanMessage):
            context_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            context_parts.append(f"Assistant: {msg.content}")
    
    context = "\n".join(context_parts) if context_parts else "No previous context."
    
    return {"context": context}


def generate_answer(state: QAState) -> QAState:
    """Generate answer using LLM with context."""
    question = state["question"]
    context = state.get("context", "")
    needs_context = state.get("needs_context", False)
    
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    # Build prompt with or without context
    if needs_context and context != "No previous context.":
        prompt = f"""Previous conversation:
{context}

Current question: {question}

Please answer the question considering the previous conversation context."""
    else:
        prompt = question
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "answer": response.content,
        "messages": state["messages"] + [HumanMessage(content=question), response]
    }


def no_context_needed(state: QAState) -> QAState:
    """Skip context retrieval when not needed."""
    return {"context": ""}


# ============ Build Graph ============
def create_qa_graph() -> StateGraph:
    """Create the LangGraph QA workflow."""
    workflow = StateGraph(QAState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_question)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("skip_context", no_context_needed)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Conditional edges
    workflow.add_conditional_edges(
        "analyze",
        lambda state: "retrieve" if state["needs_context"] else "skip_context",
        {
            "retrieve": "retrieve",
            "skip_context": "skip_context"
        }
    )
    
    # From retrieve or skip_context to generate
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("skip_context", "generate")
    
    # End at generate
    workflow.add_edge("generate", END)
    
    return workflow


# ============ Main Application ============
class ContextAwareQA:
    """Main QA system with context awareness."""
    
    def __init__(self):
        self.graph = create_qa_graph()
        # Use MemorySaver for checkpointing (conversation history)
        self.checkpointer = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        self.thread_id = str(datetime.now().timestamp())
    
    def ask(self, question: str) -> str:
        """Ask a question with automatic context handling."""
        initial_state = {
            "messages": [],
            "question": question,
            "context": "",
            "answer": "",
            "needs_context": False
        }
        
        result = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": self.thread_id}}
        )
        
        return result["answer"]
    
    def ask_with_history(self, question: str, thread_id: str = None) -> tuple[str, str]:
        """Ask a question with conversation history."""
        if thread_id:
            self.thread_id = thread_id
            
        initial_state = {
            "messages": [],
            "question": question,
            "context": "",
            "answer": "",
            "needs_context": False
        }
        
        result = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": self.thread_id}}
        )
        
        return result["answer"], self.thread_id


# ============ CLI Demo ============
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🤖 Context-Aware Q&A System (LangChain + LangGraph)")
    print("=" * 50)
    print("Type 'exit' to quit\n")
    
    qa = ContextAwareQA()
    thread = None
    
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        answer, thread = qa.ask_with_history(question, thread)
        print(f"Bot: {answer}\n")
        print(f"[Thread ID: {thread}]")
