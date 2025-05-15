import cohere
from google import genai
from redis import Redis
from redisvl.index import SearchIndex
from fastapi import FastAPI, HTTPException , UploadFile , File , Body
from typing import List
import supabase
import os
from dotenv import load_dotenv
import helper as helper
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.schema import Document
from io import BytesIO
import requests
import schema
from redisvl.query import VectorQuery


load_dotenv()


app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL")
SUPABASE_KEY = os.getenv("APIKEY_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = 'os.getenv("GEMINI_API_KEY")'

VECTOR_SCHEMA = {

    "index": {
        "name": "multimodal",
        "prefix": "rag"
    },
    "fields": [
        {"name": "content_name", "type": "text"},
        {"name": "content",      "type": "text"},
        {"name": "text",         "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 1536,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ]

}


index = SearchIndex.from_dict(VECTOR_SCHEMA, redis_url=REDIS_URL)
co = cohere.ClientV2(api_key=COHERE_API_KEY)
client_llm = genai.Client(api_key=GEMINI_API_KEY)
Client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)




@app.post("/create-schema")
async def create_schema():

    try:
        index.create(overwrite=True)
        return {
            "ok": True,
            "message": f"Index '{VECTOR_SCHEMA['index']['name']}' berhasil dibuat/di-overwrite."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    



@app.post("/upload-image")
async def upload_image(files: List[UploadFile] = File(...)):
    uploaded = []
    data = []

    for file in files:
        try:
            content = await file.read()
            remote_path = f"image/{file.filename}"
            Client.storage.from_("retrival redis").upload(remote_path, content)
            url_res = Client.storage.from_("retrival redis").get_public_url(remote_path)
            uploaded.append({
                "filename": file.filename,
                "public_url": url_res
            })
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload {file.filename}: {e}"
            )

    for file in uploaded:
        try:
            api_input_document = {
                "content": [
                    {"type": "image", "image": helper.base64_from_image(file["public_url"])}
                ]
            }
            api_response = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input_document],
            )
            emb = np.asarray(api_response.embeddings.float[0], dtype=np.float32)
            byte = emb.tobytes()
            data.append({
                "content_name": file['filename'],
                "embedding": byte,
                "content": file["public_url"],
                "text": " "
            })
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to embed {file['filename']}: {e}"
            )

    try:
        index.load(data=data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data into index: {e}"
        )

    return {
        "result": "berhasil mengupload gambar ke supabase dan embbeding",
        "uploaded": [i["filename"] for i in uploaded]
    }





@app.post("/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...))  :

    uploaded = []

    for file in files:

        try : 
            content = await file.read()
            remote_path = f"pdf/{file.filename}"
            Client.storage.from_("retrival redis").upload(remote_path, content)

            url_res = Client.storage.from_("retrival redis").get_public_url(remote_path)
            uploaded.append({
                "filename": file.filename,
                "public_url": url_res
            })
        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload {file.filename}: {e}"
            )

    for file in uploaded : 

        try : 

            data   = requests.get(file['public_url']).content
            reader = PdfReader(BytesIO(data))
            docs   = [Document(page_content=page.extract_text() or "",metadata={"source": file['public_url'], "page": i+1})for i, page in enumerate(reader.pages)]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,    
            )
            chunks = splitter.split_documents(docs)
            texts = [chunk.page_content for chunk in chunks]


            texts_emb_model = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                texts=texts,
                embedding_types=["float"],
            )

            texts_emb = texts_emb_model.embeddings.float

            final_text_data = []

            for i ,text in enumerate(texts_emb) : 
                emb = np.asarray(text , dtype=np.float32).tobytes()
                final_text_data.append({'content_name' : file['filename'] , "embedding" : emb , "content" : file['public_url'] , "text" : texts[i]})
        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Failed to embed {file['filename']}: {e}"
            )

        try:
            index.load(data=final_text_data)
        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Failed to load data into index: {e}"
            )


    return  {
        "result": "berhasil mengupload pdf ke supabase dan embbeding ",
        "uploaded": [i["filename"] for i in uploaded]
    }
   
    

@app.post("/retrival")
def retrival(request : schema.retrival): 


    try : 

        api_response = co.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[request.query],
        )

        query_emb = np.asarray(api_response.embeddings.float[0] , dtype=np.float32)

    except Exception as e:

        raise HTTPException(
                status_code=500,
                detail=f"Failed to embedding query: {e}"
            )
    

    try : 
        query = VectorQuery(
            vector=query_emb,
            vector_field_name="embedding",
            return_fields=["content_name", "content", 'embedding' , "text" ],
            num_results=12
        )
    except Exception as e:

        raise HTTPException(
                status_code=500,
                detail=f"Failed to load data into index: {e}"
            )
    
    results = index.query(query)



    return results


@app.post("/llm-generate")
def answer(query: schema.retrival):

    threshold = 0.5
    try:
        docs = retrival(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    try:
        high_score = [doc for doc in docs if float(doc.get('vector_distance', 1)) < threshold]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filtering results failed: {e}")

    content = []
    for doc in high_score:
        try:
            if doc['content_name'].lower().endswith('.pdf'):
                content.append(doc['text'])
            else:
                img_url = doc['content']
                pil_img = helper.load_image_from_url(img_url)
                content.append(pil_img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing content for {doc.get('content_name')}: {e}")

    if not content:
        content.append('no content')

    prompt = [
                f"""Answer the question based on the following image and text.
        Don't use markdown.
        Please provide enough context for your answer.

        always says what docs you retrieve

        Question: {query.query}""", content
    ]

    try:
        response = client_llm.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt
        )
        answer_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {"result": answer_text}
