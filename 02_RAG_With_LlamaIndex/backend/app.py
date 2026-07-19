import os
from dotenv import load_dotenv
import gradio as gr

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.cohere import Cohere
# ייבוא רכיב ה-Embeddings של Cohere
from llama_index.embeddings.cohere import CohereEmbedding 
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import get_response_synthesizer
from pinecone import Pinecone

from llama_index.core.workflow import (
    Workflow, 
    Event, 
    StartEvent, 
    StopEvent, 
    step, 
    Context
)
from llama_index.core.schema import NodeWithScore

load_dotenv()

# וידאו קיום המפתח
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("אנא ודאי ש-COHERE_API_KEY מוגדר בקובץ .env")

# 1. הגדרת מודל ה-LLM
Settings.llm = Cohere(
    api_key=os.getenv("COHERE_API_KEY"), 
    model="command-r-08-2024"
)

# 2. הגדרת מודל ה-Embeddings - פותר את השגיאה של OpenAI
Settings.embed_model = CohereEmbedding(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-multilingual-v3.0", # מודל מעולה התומך בעברית ובאנגלית במקביל
    input_type="search_query"
)

# אתחול החיבור ל-Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_index = pc.Index(name=index_name)

# הגדרת ה-Vector Store ויצירת ה-Index (כעת הוא ישתמש ב-Cohere Embedding במקום ב-OpenAI)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# ----------------------------------------------------
# 1. הגדרת האירועים (Events) המעודכנים
# ----------------------------------------------------
class RetrievalEvent(Event):
    """נושא את המקטעים שנשלפו ואת השאילתה המקורית"""
    nodes: list[NodeWithScore]
    query: str

class ValidationEvent(Event):
    """נושא את המקטעים המאושרים ואת השאילתה המקורית"""
    nodes: list[NodeWithScore]
    query: str


# ----------------------------------------------------
# 2. בניית ה-Workflow המעודכן ללא תלות ב-Context Dynamic Attrs
# ----------------------------------------------------
class RAGWorkflow(Workflow):
    
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievalEvent | StopEvent:
        """שלב 1: שליפת המקטעים מתוך Pinecone"""
        query = ev.get("query")
        if not query or query.strip() == "":
            return StopEvent(result="נא להזין שאלה תקינה.")
            
        # ביצוע השליפה
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        
        return RetrievalEvent(nodes=nodes, query=query)

    @step
    async def validate(self, ctx: Context, ev: RetrievalEvent) -> ValidationEvent | StopEvent:
        """שלב 2: ולידציות, בדיקות תקינות וסינון סמנטי"""
        nodes = ev.nodes
        query = ev.query
        
        if not nodes:
            return StopEvent(result="לא נמצא מידע רלוונטי במסמכים עבור שאלה זו.")
            
        filtered_nodes = [node for node in nodes if node.score >= 0.5]
        
        if not filtered_nodes:
            return StopEvent(result="לא נמצא מידע בעל רמת מובהקות מספקת כדי לענות על השאלה.")
            
        return ValidationEvent(nodes=filtered_nodes, query=query)

    @step
    async def synthesize(self, ctx: Context, ev: ValidationEvent) -> StopEvent:
        """שלב 3: ניסוח התשובה הסופית בעזרת ה-LLM"""
        query = ev.query
        filtered_nodes = ev.nodes
        
        # שימוש ב-Synthesizer הרשמי
        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize(query=query, nodes=filtered_nodes)
        
        return StopEvent(result=str(response))
# אתחול ה-Workflow
rag_workflow = RAGWorkflow(timeout=30.0, verbose=True)

# ----------------------------------------------------
# 3. ממשק ה-Gradio
# ----------------------------------------------------

async def RAG_chat_interface(message, history):
    try:
        # הרצת ה-Workflow בצורה אסינכרונית
        result = await rag_workflow.run(query=message)
        return result
    except Exception as e:
        return f"התרחשה שגיאה במהלך הרצת ה-Workflow: {str(e)}"

# בניית ממשק המשתמש באמצעות Gradio (גרסה מעודכנת)
with gr.Blocks(title="AI Developer Agent - Event-Driven RAG") as demo:
    gr.Markdown("# 🤖 Event-Driven AI Developer Agent")
    gr.Markdown("צינור RAG מבוסס שלבים ואירועים (LlamaIndex Workflows).")
    
    gr.ChatInterface(
        fn=RAG_chat_interface,
        textbox=gr.Textbox(placeholder="הקלידי את השאלה שלך כאן...", container=False, scale=7)
    )

# הרצת השרת
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
