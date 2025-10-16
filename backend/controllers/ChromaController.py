from backend.services.ChromaService import ChromaService

class ChromaController:
    def __init__(self):
        self.chromaService = ChromaService()
    def request_store_pdf(self):
        pdf_path = input("Please enter the file path of the pdf: ")

        self.chromaService.store_file_chroma(pdf_path)

