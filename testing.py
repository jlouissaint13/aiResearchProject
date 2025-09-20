import sys

from flask import Flask
import pymupdf4llm
import pathlib
from ChunkSplitter import ChunkSplitter
from ChromaManager import ChromaManager
from EmbedHandler import EmbedHandler
from Rag import Rag
chroma = ChromaManager()
chunkSplitter = ChunkSplitter()
embedHandler = EmbedHandler()



def pdf_to_text():
    pdf_path  = input("Please enter the file path of the pdf: ")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    base_directory = pathlib.Path(__file__).parent
    output_directory = base_directory / "pdfToText"

    pathlib.Path(output_directory / "result.md").write_text(md_text,encoding="utf-8")

    return md_text

def test_chunk_embedding(text):
    chunks = chunkSplitter.semantic_split(text)

    embedHandler.store_chunks_and_embeddings(chunks)

def test_database():

    chroma.check_db()

def question():
    q = "y"
    while q == "yes" or q == "y":
        user_question = input("What is your question? ")
        rag = Rag(chroma,user_question)
        print(rag.model_run())
        q = input("Would you like to ask another question?(Y/N) ").lower()

if __name__ == '__main__':
    user_input = -1

    while user_input != 6:
        try:
            print("Please make a selection\n1)Ask a question\n2)Insert a pdf\n3)Check the database\n4)Delete the database\n5)Choose your model(Work In Progress)\n6)Quit")
            user_input = int(input())


            match user_input:
                case 1: question()
                case 2: test_chunk_embedding(pdf_to_text()) #pdf_to_text()
                case 3: chroma.check_db()
                case 4: chroma.delete()
                case 5: sys.exit(1)
                case 6: sys.exit(0)
                case _: print("Invalid selection")
        except ValueError: print("Please Enter A Numerical Value")





