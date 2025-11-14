import hashlib
from pypdf import PdfReader
from backend.models.Metadata import Metadata as metadata_dto

class MetadataServie:
    def __init__(self,pdf_path):
        self.reader = PdfReader(pdf_path)
        self.pdf_path = pdf_path
        self.metadata_dto = metadata_dto
    def get_metadata_dto(self,file_name):
        self.metadata_dto.title =  str(self.reader.metadata.title)
        self.metadata_dto.author = str(self.reader.metadata.author)
        self.metadata_dto.created_at = str(self.reader.metadata.creation_date)
        self.metadata_dto.modified_at = str(self.reader.metadata.modification_date)
        self.metadata_dto.hash_value = str(self.hash_pdf())
        self.metadata_dto.file_name = str(file_name)
        return self.metadata_dto

    def hash_pdf(self):
        hash_func = hashlib.new('sha256')
        with open(self.pdf_path,'rb') as f:
            while True:

                data = f.read(131072) #buffer size

                if not data:
                    break

                hash_func.update(data)
        return hash_func.hexdigest()




