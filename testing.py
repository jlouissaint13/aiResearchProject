import sys
import time
from backend.repository.ChromaRepository import ChromaRepository
from setup.Startup import Startup
from backend.services.RagService import RagService

chroma = ChromaRepository()
startup = Startup()

if __name__ == '__main__':
    rag = None
    user_input = -1

    while user_input != 6:
        try:
            print("Please make a selection:\n"
                  "1) Ask a question\n"
                  "2) Show most relevant results\n"
                  "3) Insert a pdf\n"
                  "4) Enter API Key \n"
                  "5) Check the database\n"
                  "6) Delete the database\n"
                  "7) Choose your model (Work In Progress)\n"
                  "8) Quit")
            user_input = int(input())


            match user_input:
                case 1:
                    if rag is None:
                        rag = RagService()
                    rag.response("What is the pdf about?")
                case 2:
                    pass
                case 3:
                  pass
                case 5:
                    chroma.check_db()
                case 6:
                    chroma.delete_database()
                case 7:
                    sys.exit(1)
                case 8:
                    sys.exit(0)
                case _:
                    print("Invalid selection")

            end_time = time.time()  # Stop the stopwatch

        except ValueError:
            print("Please Enter A Numerical Value")




