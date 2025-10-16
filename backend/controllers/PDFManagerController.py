from flask import jsonify,request,Blueprint


pdf_blueprint = Blueprint('pdf_manager',__name__)


@pdf_blueprint.route("/insert_pdf")
def insert_pdf():
    pass