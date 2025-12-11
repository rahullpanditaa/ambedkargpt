from src.ingest.pdf_ingest import PDFIngestion

def main():
    pi = PDFIngestion()
    pi._extract_paragraphs()
    pi._extract_sentences()


if __name__ == "__main__":
    main()
