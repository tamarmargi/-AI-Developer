import os
from dotenv import load_dotenv  
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.cohere import CohereEmbedding

# מציאת נתיב תיקיית השורש של הפרויקט באופן דינמי
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "../../../"))

# טעינת קובץ ה-.env ישירות מתיקיית השורש של הפרויקט
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

def load_parse_and_embed_documents():
    # --- שלב 1: טעינה ---
    print("🤖 שלב 1: טעינת מסמכים...")
    target_dir = os.path.join(project_root, "agent_docs")
    
    reader = SimpleDirectoryReader(input_dir=target_dir, recursive=True, required_exts=[".md"])
    documents = reader.load_data()
    print(f"✅ נטענו {len(documents)} מסמכים.")

    # --- שלב 2: פיצול (Chunking) ---
    print("\n🤖 שלב 2: חלוקת המסמכים למקטעים...")
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"✅ נוצרו {len(nodes)} מקטעים (Nodes).")

    # --- שלב 3: יצירת ה-Embeddings עם Cohere ---
    print("\n🤖 שלב 3: התחברות ל-Cohere ויצירת Embeddings...")
    
    # וידוא ידני שהמפתח נטען כראוי למערכת למניעת שגיאות Pydantic
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError(f"שגיאה: COHERE_API_KEY לא נמצא! חיפשתי בקובץ: {dotenv_path}")
        
    embed_model = CohereEmbedding(
        model_name="embed-multilingual-v3.0",
        api_key=cohere_key
    )
    
    # בדיקת תקינות התקשורת
    if nodes:
        sample_node = nodes[0]
        print(f"מריץ בדיקת קישוריות ל-Cohere עבור המקטע מתוך: {sample_node.metadata.get('file_name')}")
        
        sample_vector = embed_model.get_text_embedding(sample_node.get_content())
        
        print("✅ התקבל מענה מ-Cohere בהצלחה!")
        print(f"📏 אורך הוקטור (Dimension): {len(sample_vector)}")
        print(f"🔢 דוגמה למספרים הראשונים בוקטור: {sample_vector[:5]}")
        
    return nodes, embed_model

if __name__ == "__main__":
    load_parse_and_embed_documents()