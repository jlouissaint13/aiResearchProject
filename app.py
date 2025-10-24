from flask import Flask
from flask_cors import CORS

from backend.controllers.ConversationController import conversation_blueprint
from backend.controllers.LoginController import login_blueprint
from backend.controllers.MessageController import message_blueprint
from backend.controllers.RagController import rag_blueprint
from backend.controllers.RegisterController import register_blueprint
from backend.controllers.UserSettingsController import user_settings_blueprint
from backend.controllers.PDFManagerController import pdf_blueprint
from backend.controllers.ModelSettingsController import model_settings_blueprint
from backend.controllers.StoreModelController import store_model_blueprint
from backend.controllers.OpenAIController import openai_blueprint
from backend.controllers.GeminiController import gemini_blueprint
from backend.setup.setup_env import setup_env
app = Flask(__name__)




app.register_blueprint(rag_blueprint,url_prefix='/rag')
app.register_blueprint(register_blueprint,url_prefix='/user')
app.register_blueprint(login_blueprint,url_prefix="/login")
app.register_blueprint(conversation_blueprint,url_prefix="/conversation")
app.register_blueprint(message_blueprint,url_prefix="/message")
app.register_blueprint(user_settings_blueprint,url_prefix="/user_settings")
app.register_blueprint(pdf_blueprint,url_prefix="/pdf_manager")
app.register_blueprint(model_settings_blueprint,url_prefix="/model_settings")

app.register_blueprint(store_model_blueprint,url_prefix="/store_model")

app.register_blueprint(openai_blueprint,url_prefix="/open_ai_api")

app.register_blueprint(gemini_blueprint,url_prefix="/gemini_api")
CORS(app,origins="*")

if __name__ == '__main__':
    setup_env()
    app.run(port=8000,debug=True)
