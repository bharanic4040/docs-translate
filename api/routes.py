import time

from fastapi import APIRouter
from paddleocr import PaddleOCR
from translate.custom_translation_hugging import translate

ocr = PaddleOCR(lang="ta",
    text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="ta_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False, use_doc_unwarping=False,use_textline_orientation=False,)

doc_path = "files/story.png"

router = APIRouter()

@router.get("/translate")
async def translate_document():
    try:
        start_time = time.time()
        result = ocr.predict(doc_path)
        full_text = ""
        input_translated_list = []
        for page in result:
            texts = page['rec_texts']
            for text in texts:
                full_text += text + "\n"
            input_translated_list.append(full_text)
        # tel_Telu, tam_Taml, hin_Deva
        output_target = translate(input_translated_list, src_lang="tam_Taml", target_lang="tel_Telu")
        print("Total time in secs: ", time.time() - start_time)
    except Exception as e:
        return {"Error": str(e)}

    return { "translation": output_target }

@router.get("/health")
def health():
    return {"status": "ok"}