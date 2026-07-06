import tempfile
import time

from fastapi import APIRouter, UploadFile, File
from paddleocr import PaddleOCR
from translate.custom_translation_hugging import translate

ocr = PaddleOCR(
    lang="ta",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="ta_PP-OCRv5_mobile_rec",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
doc_path = "files/story.png"

router = APIRouter()

@router.post("/translate")
async def translate_document(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        image_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp.write(await file.read())
            image_path = temp.name
        result = ocr.predict(image_path)
        all_page_translations = []
        for page in result:
            texts = page['rec_texts']
            max_chars = 128
            chunks = []
            current = ""
            for line in texts:
                if len(current) + len(line) > max_chars:
                    chunks.append(current)
                    current = line
                else:
                    current += "\n" + line
            if current:
                chunks.append(current)
            translated = translate(
                chunks, src_lang="tam_Taml", target_lang="tel_Telu"
            )
            final_translation = ".".join(translated)
            #TODO - IMP - need to strip backward slashes and double quotes
            all_page_translations.append(final_translation)
        print("Total time in secs: ", time.time() - start_time)
    except Exception as e:
        return {"Error": str(e)}

    return { "translation":  all_page_translations }

@router.get("/health")
def health():
    return {"status": "ok"}