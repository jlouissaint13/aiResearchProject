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
    
    file_path = pdf_service.local_store_pdf(file_path)    
           
    if pdf_service.store_pdf_database(file_path, file_name, user_id) == 409:
        return "File already exists for this user" , 409


    #chroma service will go last so I can insert the right file path and do the check on line 18 for while existence on a specific user
    chroma_service.store_pdf_chroma(file_path, file_name,user_id)





    return "success" ,200


@pdf_blueprint.route('/retrieve_all_pdfs',methods=['POST'])
def retrieve_pdfs():
    data = request.get_json()
    user_id = data.get("user_id")
    response = pdf_service.pdf_storage_retrieve_pdfs(user_id)
    
    return jsonify(response) , 200
    


@pdf_blueprint.route("/delete_pdf",methods=['Delete'])
def delete_pdf():
    data = request.get_json()
    user_id = data.get('user_id')
    pdf_path = data.get("file_path")
    
    hash_value = str(pdf_service.retrieve_hash_value(pdf_path))
    
    pdf_service.delete_pdf_from_database(user_id, pdf_path)    
    
  
    chroma_service.delete_from_chroma_db(hash_value)


    pdf_service.delete_pdf_from_local_storage(pdf_path,hash_value)
    
    
    
    
    return "success", 200
  
  
  

    