
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
    full_text = ""
    input_translated_list = []
    for page in result:
        texts = page['rec_texts']
        for text in texts:
            full_text += text + "\n"
        input_translated_list.append(full_text)
    print(full_text)
    # tel_Telu, tam_Taml, hin_Deva, kan_Knda, mal_Mlym
    output_target = translate(input_translated_list, src_lang="tam_Taml", target_lang="tel_Telu")
    for s in output_target:
        print(s)


if __name__ == "__main__":
    main()
