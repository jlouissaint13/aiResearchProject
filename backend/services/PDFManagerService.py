import shutil

from backend.repository.PDFManagerRepository import PDFManagerRepository
from backend.services.MetadataService import MetadataServie
from pathlib import Path

class PDFManagerService:
    def __init__(self):
        self.metadata_service = None
        self.pdf_manager_repository = PDFManagerRepository()
        
        
    def store_pdf_database(self, pdf_path, pdf_name, user_id):
        self.metadata_service = MetadataServie(pdf_path)
        hash_value = self.metadata_service.hash_pdf()
        
        pdf_path = str(pdf_path)
        
        self.pdf_manager_repository.insert_pdf(pdf_name,user_id,hash_value,pdf_path)
            
    def local_store_pdf(self,pdf_path):
        base_dir = Path(__file__).resolve().parent.parent.parent
        
        pdf_folder = base_dir/"pdfstorage"
        pdf_folder.mkdir(exist_ok=True)
        pdf_path = Path(pdf_path)
        
        location = pdf_folder / pdf_path.name
        
        shutil.copy(pdf_path,location)
        
        return location
    
    
    def pdf_storage_retrieve_pdfs(self,user_id):
       return self.pdf_manager_repository.retrieve_all_pdfs_by_user_id(user_id)
    
    
    
    
    def delete_pdf_from_database(self,user_id,file_path):
        self.metadata_service = MetadataServie(file_path)
        hash_value = self.metadata_service.hash_pdf()
        self.pdf_manager_repository.delete_pdf_by_user_id_and_hash_value(user_id,hash_value)
        return 200
         
        
    def delete_pdf_from_local_storage(self,file_path):
        file = Path(file_path)
        if file.exists():
            file.unlink()
        