from pathlib import Path
from pdfminer.high_level import extract_text

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
AMBEDKAR_BOOK_PATH =  DATA_DIR_PATH / "Ambedkar_book.pdf"

RAW_BOOK_TEXT = extract_text(AMBEDKAR_BOOK_PATH)


class PDFIngestion:
    def __init__(self, pdf=RAW_BOOK_TEXT):
        self.pdf_text = pdf
        self.pages = []

    def _split_pdf_into_pages(self):
        # split on form feed characters
        pages = self.pdf_text.split("\x0c")
        self.pages = pages