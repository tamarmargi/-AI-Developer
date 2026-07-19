import os
from typing import List
from dotenv import load_dotenv

# ייבוא מהקובץ בעל השם הייחודי - פותר את בעיית ההתנגשות לחלוטין
from project_schemas import ProjectStructuredData

from llama_index.llms.cohere import Cohere
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core import SimpleDirectoryReader

load_dotenv()

def extract_structured_data_from_docs(directory_path: str) -> ProjectStructuredData:
    """
    פונקציה הסורקת תיקיית מסמכים ומחלצת מתוכם מידע מובנה 
    לפי סכמת ההחלטות, הכללים והאזהרות.
    """
    # 1. טעינת המסמכים מתוך התיקייה (קבצי ה-md)
    reader = SimpleDirectoryReader(
        input_dir=directory_path, 
        required_exts=[".md"],
        recursive=True
    )
    documents = reader.load_data()
    
    if not documents:
        print("לא נמצאו קבצי תיעוד (md) בתיקייה המבוקשת.")
        return ProjectStructuredData()

    # איחוד תוכן המסמכים לצורך ניתוח מקיף
    combined_content = "\n\n".join([doc.text for doc in documents])

    # 2. הגדרת מודל השפה הייעודי לחילוץ מובנה
    llm = Cohere(api_key=os.getenv("COHERE_API_KEY"), model="command-r-08-2024")

    # 3. הגדרת ה-Prompt שמנחה את המודל כיצד לזהות ולמפות את הפריטים
    prompt_template_str = """
    הנך אנליסט מערכות מומחה. תפקידך הוא לסרוק את תיעוד הפרויקט הבא ולחלץ מתוכו מידע מובנה בצורה מדויקת על פי ההנחיות.
    
    עליך למצוא ולחלץ את סוגי הפריטים הבאים:
    1. Decisions: החלטות ארכיטקטורה וטכנולוגיה חשובות שהתקבלו.
    2. Rules: כללי פיתוח, הנחיות קוד קשיחות או דרישות ממשק (כמו RTL, ניהול סטייט).
    3. Warnings: אזהרות קריטיות, נקודות רגישות במערכת או רכיבים שאסור לשנות ללא בדיקה.
    
    עבור כל פריט, נסה להעריך בצורה הטובה ביותר את טווח השורות (line_range) ושם הקובץ הרלוונטי מתוך הטקסט במידת האפשר.
    
    תוכן התיעוד המלא של הפרויקט:
    ---------------------------------
    {document_text}
    ---------------------------------
    
    חלץ את כל המידע המובנה התואם את הפורמט הנדרש באופן מלא ומקיף.
    """

    # 4. יצירת תוכנית החילוץ המובנית של LlamaIndex
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ProjectStructuredData,
        prompt_template_str=prompt_template_str,
        llm=llm,
        verbose=True
    )

    print("מתחיל בתהליך חילוץ הנתונים מהמסמכים...")
    # הרצת תוכנית החילוץ
    structured_output = program(document_text=combined_content)
    print("החילוץ הושלם בהצלחה!")
    
    return structured_output

# ----------------------------------------------------
# הרצה עצמאית לבדיקה ושמירת התוצאות לקובץ JSON
# ----------------------------------------------------
if __name__ == "__main__":
    # נתיב לתיקיית הדוקומנטציה של הפרויקט שלך
    docs_dir = "../../agent_docs"    
    if os.path.exists(docs_dir):
        result = extract_structured_data_from_docs(docs_dir)
        
        # שמירת הפלט המובנה כקובץ JSON מסודר
        output_json_path = "project_data.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(result.json(indent=2, ensure_ascii=False))
            
        print(f"הנתונים המובנים נשמרו בהצלחה בקובץ: {output_json_path}")
    else:
        print(f"אנא ודאי שקיימת תיקייה בשם '{docs_dir}' המכילה את קבצי ה-md שלך.")