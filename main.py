from src.ingest.pdf_ingest import PDFIngestion

def main():
    pdf_ingestor = PDFIngestion()
    pdf_ingestor._extract_paragraphs()


if __name__ == "__main__":
    main()
