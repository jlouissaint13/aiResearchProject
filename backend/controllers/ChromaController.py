from backend.services.ChromaService import ChromaService
from flask import Blueprint,request



chroma_blueprint = Blueprint("chroma",__name__)


  
chromaService = ChromaService()


@chroma_blueprint.route()
def request_store_pdf():
   # pdf_path = input("Please enter the file path of the pdf: ")

    #chromaService.store_file_chroma(pdf_path)
    pass
