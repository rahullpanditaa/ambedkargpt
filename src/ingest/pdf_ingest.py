from pathlib import Path
from pdfminer.high_level import extract_text

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
AMBEDKAR_BOOK_PATH =  DATA_DIR_PATH / "Ambedkar_book.pdf"

RAW_BOOK_TEXT = extract_text(AMBEDKAR_BOOK_PATH)


class PDFIngestion:
    def __init__(self, pdf=RAW_BOOK_TEXT):
        self.pdf_text = pdf

        # each str is an entire page's text
        self.pages: list[str] = []

    def _split_pdf_into_pages(self):
        # split on form feed characters
        pages = self.pdf_text.split("\x0c")
        self.pages = pages

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
        
        return paragraphs


