import json
import spacy
from src.utils.constants import (
    RAW_BOOK_TEXT,
    PROCESSED_DATA_DIR_PATH,
    BOOK_PARAGRAPHS_PATH,
    BOOK_SENTENCES_PATH
)

class PDFIngestion:
    """
    Handles ingestion and structural decomposition of a raw PDF document.

    Responsibilities:
    - Split raw PDF text into pages
    - Extract paragraph-level units with page provenance
    - Extract sentence-level units with stable global IDs
    - Persist intermediate artifacts for downstream pipeline stages
    """
    def __init__(self, pdf=RAW_BOOK_TEXT):
        """
        Initialize the ingestion pipeline with raw PDF text.

        Args:
            pdf (str): Full raw text of the PDF. Pages are expected to be
                       delimited by the form-feed character ('\\x0c').

        Attributes:
            pages (list[str]): Cleaned page-level text extracted from the PDF
            paragraphs (list[dict]): Paragraph metadata populated after extraction
            sentences (list[dict]): Sentence metadata populated after extraction
        """

        # each str is an entire page's text
        self.pages: list[str] = [p for p in pdf.split("\x0c") if p.strip() != ""]
        # self.paragraphs: list[dict] = []
        # self.sentences = []

    # each page of text -> list of paragraphs
    def _extract_paragraphs(self) -> list[dict]:
        """
        Extract paragraph-level units from each page of the document.

        Paragraphs are detected using blank lines as boundaries.
        Each paragraph retains provenance metadata including:
        - Page number
        - Paragraph index within the page

        Returns:
            list[dict]: List of paragraph objects with schema:
                {
                    "page": int,
                    "para_idx": int,
                    "text": str
                }

        Side Effects:
            - Creates the processed data directory if it does not exist
            - Writes paragraph data to BOOK_PARAGRAPHS_PATH
        """
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
                current_paragraph_lines.append(cleaned_line)

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
        # self.paragraphs = paragraphs
        return paragraphs
    
    def extract_sentences(self):
        """
        Split extracted paragraphs into sentence-level units.

        Uses spaCy's lightweight sentencizer (rule-based) to avoid
        unnecessary NLP overhead. Each sentence is assigned a stable
        global ID and retains paragraph and page provenance.

        Returns:
            list[dict]: List of sentence objects with schema:
                {
                    "id": str,
                    "page": int,
                    "para_idx": int,
                    "sentence_idx": int,
                    "text": str
                }

        Side Effects:
            - Writes sentence data to BOOK_SENTENCES_PATH

        Preconditions:
            - _extract_paragraphs must be called first
        """
        paragraphs = self._extract_paragraphs()
        sentences = []
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        global_id = 1
        for paragraph in paragraphs:
            doc = nlp(paragraph["text"])
            sentence_idx = 0
            for sentence in doc.sents:                
                s = " ".join(sentence.text.strip().split())
                if s == "":
                    continue
                sentences.append({
                    "id": f"sent_{global_id:06}",
                    "page": paragraph["page"],
                    "para_idx": paragraph["para_idx"],
                    # sentence index per paragraph
                    "sentence_idx": sentence_idx,
                    "text": s
                })
                global_id += 1
                sentence_idx += 1

        with open(BOOK_SENTENCES_PATH, "w") as f:
            json.dump({
                "document": "Ambedkar_book.pdf",
                "sentences": sentences},
                f, indent=2)
        # self.sentences = sentences
        return sentences