# AI-Developer-Agent (RAG System with LlamaIndex & Pinecone)

מערכת RAG (Retrieval-Augmented Generation) מתקדמת המאפשרת תשאול אינטליגנטי בשפה חופשית על גבי מסמכי תיעוד טכניים (Markdown). המערכת משלבת בסיס נתונים ווקטורי לחיפוש סמנטי, סינון מקטעים לא רלוונטיים, ומודל שפה מוביל להפקת תשובות מדויקות.

## 🚀 תכונות מרכזיות (Features)

- **חיפוש סמנטי חוצה שפות:** אפשרות לשאול שאלות בעברית ולקבל מידע מדויק מתוך מסמכי מקור הכתובים באנגלית.
- **בסיס נתונים ווקטורי (Pinecone):** שמירה ואחזור מהירים של ה-Embeddings של מסמכי התיעוד.
- **מנגנון Postprocessing מתקדם:** שימוש ב-`SimilarityPostprocessor` עם רף חסימה דינמי (`similarity_cutoff=0.5`) כדי לסנן מקטעי מידע בעלי דמיון נמוך ולמנוע הזיות (Hallucinations).
- **ניהול מצבי קצה:** טיפול מובנה במצבים בהם לא נמצא מידע רלוונטי, תוך החזרת הודעה ברורה למשתמש במקום תגובה ריקה.
- **ממשק משתמש אינטראקטיבי:** צ'אט בוט מבוסס **Gradio** המאפשר עבודה חלקה ונוחה מול המערכת.

## 🏗️ ארכיטקטורת המערכת

המערכת בנויה מצינור ה-RAG הבא:
1. **Data Ingestion:** טעינת מסמכי ה-Markdown ופירוקם למקטעים (Nodes).
2. **Embedding & Vector Store:** הפיכת הטקסט לווקטורים סמנטיים ואחסונם ב-Pinecone.
3. **Retrieval:** שליפת 5 המקטעים המובילים (`similarity_top_k=5`) בעת קבלת שאילתה.
4. **Postprocessing:** סינון אקטיבי של המקטעים לפי ציון הדמיון שלהם לפרומפט.
5. **Synthesis:** כיווץ המידע שעבר את הסינון (`response_mode="compact"`) ושליחתו למודל השפה.
6. **LLM Generation:** הפקת התשובה הסופית באמצעות מודל `command-r-08-2024` של Cohere.

## 🛠️ טכנולוגיות בשימוש (Tech Stack)

- **Framework:** LlamaIndex
- **LLM Provider:** Cohere (`command-r-08-2024`)
- **Vector Database:** Pinecone
- **UI Framework:** Gradio
- **Language:** Python

## 💻 התקנה והרצה

1. התקנת חבילות הדרישות:
   ```bash
   pip install llama-index llama-index-llms-cohere llama-index-vector-stores-pinecone gradio python-dotenv
   
   הגדרת משתני הסביבה בקובץ .env:

COHERE_API_KEY=your_cohere_key

PINECONE_API_KEY=your_pinecone_key

הרצת השרת:
python backend/app.py
