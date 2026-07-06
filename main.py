# from pdf2image import convert_from_path
from paddleocr import PaddleOCR
# from utils.indic_translation import translate
from translate.custom_translation_hugging import translate

doc_path = "files/my-reg-tamil.png"


def main():
    # ocr = PaddleOCR(lang='te') #telugu input
    ocr = PaddleOCR(lang='ta')  # tamil input
    result = ocr.predict(doc_path)
    full_text = ""
    for page in result:
        texts = page['rec_texts']
        input_translated_list = []
        for text in texts:
            input_translated_list.append(text)
            full_text += text + "\n"
    print(full_text)
    # tel_Telu, tam_Taml
    output_target = translate(input_translated_list, src_lang="tam_Taml", target_lang="tel_Telu")
    for s in output_target:
        print(s)


if __name__ == "__main__":
    main()
