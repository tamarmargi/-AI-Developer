import os
import ssl
import urllib3
import gradio as gr
from dotenv import load_dotenv
from pinecone import Pinecone
# 1. ייבויי ליבה מתוך llama_index.core
from llama_index.core import VectorStoreIndex, Settings
# 2. ייבוא ה-PromptTemplate מתוך תיקיית ה-prompts
from llama_index.core.prompts import PromptTemplate
# 3. ייבוא ה-SimilarityPostprocessor מתוך תיקיית ה-postprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor

# --- עקיפת הגדרות SSL עבור סביבת נטפרי ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# טעינת משתני סביבה
load_dotenv()

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere  # <--- ייבוא של ה-LLM של Cohere

# 1. הגדרת מודל ה-Embedding של Cohere
Settings.embed_model = CohereEmbedding(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-multilingual-v3.0",
)

# 2. הגדרת מודל ה-LLM של Cohere (כדי שלא יחפש את OpenAI)
# הגדרת מודל ה-LLM העדכני והפעיל של Cohere
Settings.llm = Cohere(
    api_key=os.getenv("COHERE_API_KEY"),
    model="command-r-08-2024"  # <--- שינוי לשם המודל הרשמי והזמין
)

# 3. התחברות ל-Pinecone וטעינת האינדקס הקיים
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# כעת זה יעבוד ללא דרישה ל-OpenAI
# עדכון מנוע השאילתות עם פתרון מובנה לתשובות ריקות ורף מאוזן יותר
# הגדרת מנוע השאילתות עם Response Synthesizer מוגדר מראש
query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    # הגדרת אסטרטגיית הניסוח (ניתן לשנות ל-'refine' או 'tree_summarize' בהתאם לצורך)
    response_mode="compact" 
)
# פונקציית הליבה של הצ'אט שמקבלת את השאלה ומחזירה תשובה מה-LLM
# def RAG_chat_interface(user_question, history):
    # if not user_question.strip():
    #     return "בבקשה תזיני שאלה תקינה."
    
    # try:
    #     # מאחורי הקלעים: LlamaIndex הופכת את השאלה לוקטור -> שולפת מ-Pinecone -> שולחת ל-LLM
    #     response = query_engine.query(user_question)
    #     return str(response)
    # except Exception as e:
    #     return f"שגיאה במהלך השליפה או הניסוח: {str(e)}"
def RAG_chat_interface(message, history):
    # ביצוע השאילתה מול המנוע
    response = query_engine.query(message)
    
    # בדיקה האם התשובה ריקה או שכל ה-Nodes סוננו
    if not response or not response.response or response.response.strip() == "":
        return "לא נמצא מידע רלוונטי במסמכים עבור שאלה זו."
        
    return response.response
# 4. יצירת ממשק המשתמש הגרפי עם Gradio
with gr.Blocks(title="AI Developer Agent - RAG Chat") as demo:
    gr.Markdown("# 🤖 צ'אט בוט מבוסס מסמכים (RAG)")
    gr.Markdown("שאלי שאלות על בסיס מסמכי ה-Markdown המאונדקסים ב-Pinecone.")
    
    # רכיב הצ'אט המובנה של Gradio
    # gr.ChatInterface(
    #     fn=RAG_chat_interface,
    #     textbox=gr.Textbox(placeholder="הקלידי את השאלה שלך כאן...", container=False, scale=7),
    #     submit_btn="שלח 🚀",
    #     retry_btn="נסה שוב 🔄",
    #     undo_btn="מחק אחרון ↩️",
    #     clear_btn="נקה הכל 🗑️",
    # )
    # רכיב הצ'אט המעודכן והתואם לגרסאות החדשות של Gradio
    gr.ChatInterface(
        fn=RAG_chat_interface,
        textbox=gr.Textbox(placeholder="הקלידי את השאלה שלך כאן...", container=False, scale=7)
    )

# הרצת השרת המקומי
if __name__ == "__main__":
    # השרת ירוץ בכתובת http://127.0.0.1:7860
    demo.launch(share=False)