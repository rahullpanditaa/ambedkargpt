import json
import spacy
from pathlib import Path
from pdfminer.high_level import extract_text

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"

AMBEDKAR_BOOK_PATH =  DATA_DIR_PATH / "Ambedkar_book.pdf"
RAW_BOOK_TEXT = extract_text(AMBEDKAR_BOOK_PATH)

PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"

BOOK_PARAGRAPHS_PATH = PROCESSED_DATA_DIR_PATH / "paragraphs.json"
BOOK_SENTENCES_PATH = PROCESSED_DATA_DIR_PATH / "sentences.json"

class PDFIngestion:
    def __init__(self, pdf=RAW_BOOK_TEXT):

        # each str is an entire page's text
        self.pages: list[str] = pdf.split("\x0c")
        self.paragraphs: list[dict] = []
        self.sentences = []

    # each page of text -> list of paragraphs
    def _extract_paragraphs(self) -> list[dict]:
        paragraphs = []
        for page_number, page_text in enumerate(self.pages, 1):

            # split page into lines
            lines = page_text.split("\n")

            current_paragraph_lines = []
            para_idx = 0

            for line in lines:
                cleaned_line = line.strip()

                if cleaned_line == "":
                    # if line is blank -> it is a para boundary
                    if current_paragraph_lines:
                        merged_text = " ".join(current_paragraph_lines).strip()
                        paragraphs.append({
                            "page": page_number,
                            "para_idx": para_idx,
                            "text": merged_text
                        })
                        para_idx += 1
                        current_paragraph_lines = []

                    continue

                # non empty line - same para
                current_paragraph_lines.append(line)

            if current_paragraph_lines:
                merged_text = " ".join(current_paragraph_lines).strip()
                paragraphs.append({
                    "page": page_number,
                    "para_idx": para_idx,
                    "text": merged_text
                })
        
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        with open(BOOK_PARAGRAPHS_PATH, "w") as f:
            json.dump({"paragraphs": paragraphs}, f, indent=2)
        self.paragraphs = paragraphs
        return paragraphs
    
    def _extract_sentences(self):
        if self.paragraphs is None:
            self._extract_paragraphs()

        sentences = []
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        id = 1
        for paragraph in self.paragraphs:
            doc = nlp(paragraph["text"])
            sentence_idx = 0
            for sentence in doc.sents:                
                s = sentence.text.strip()
                if s == "":
                    continue
                sentences.append({
                    "id": f"sent_{id:05}",
                    "page": paragraph["page"],
                    "para_idx": paragraph["para_idx"],
                    # sentence index per paragraph
                    "sentence_idx": sentence_idx,
                    "text": paragraph["text"]
                })
                id += 1
                sentence_idx += 1
            sentence_idx = 0

        with open(BOOK_SENTENCES_PATH, "w") as f:
            json.dump({
                "document": "Ambedkar_book.pdf",
                "sentences": sentences},
                f, indent=2)
        self.sentences = sentences
        return sentences