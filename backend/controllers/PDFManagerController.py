from flask import jsonify,request,Blueprint

from backend.services.ChromaService import ChromaService
from backend.services.PDFManagerService import PDFManagerService

pdf_blueprint = Blueprint('pdf_manager',__name__)
chroma_service = ChromaService()
pdf_service = PDFManagerService()
@pdf_blueprint.route("/insert_pdf",methods=['POST'])
def insert_pdf():
    data = request.get_json()
    file_path: str = data.get("file_path")
    file_name = data.get("file_name")
    user_id = data.get("user_id")
   # if chroma_service.store_pdf_chroma(file_path, file_name) != 200:
       # return "File already exists" , 409
    
    
    
    #this function will take the original file path and replace it with a new file path stored in the directory of the project
    #this way if I need to delete any files I can do it without worrying about the user changing the location of their file
    #the hash which tells us which file to delete is dependent on the file path so this could be problematic
    file_path = pdf_service.local_store_pdf(file_path)
    
    
    pdf_service.store_pdf_database(file_path, file_name, user_id)
    
    
    
    
    return "success" ,200


@pdf_blueprint.route('/retrieve_all_pdfs',methods=['POST'])
def retrieve_pdfs():
    data = request.get_json()
    user_id = data.get("user_id")
    response = pdf_service.pdf_storage_retrieve_pdfs(user_id)
    print(response)
    return jsonify(response) , 200



@pdf_blueprint.route("/delete_pdf",methods=['Delete'])
def delete_pdf():
    data = request.get_json()
    user_id = data.get('user_id')
    pdf_path = data.get("file_path")
    pdf_service.delete_pdf_from_database(user_id, pdf_path)
    pdf_service.delete_pdf_from_local_storage(pdf_path)
    return "success", 200
  
  
  

    