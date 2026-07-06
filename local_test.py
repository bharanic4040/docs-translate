
from paddleocr import PaddleOCR
from translate.custom_translation_hugging import translate

doc_path = "files/story.png"
# ocr = PaddleOCR(lang='te') #telugu input
ocr = PaddleOCR(
    lang="ta",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="ta_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def main():
    result = ocr.predict(doc_path)
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
        final_translation = "\n".join(translated)
        print(final_translation)
    # tel_Telu, tam_Taml, hin_Deva, kan_Knda, mal_Mlym


if __name__ == "__main__":
    main()
