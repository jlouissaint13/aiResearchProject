from flask import Flask
import pymupdf4llm
import pathlib
app = Flask(__name__)


def pdf_to_text():
    pdf_path  = input("Please enter the file path of the pdf")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    base_directory = pathlib.Path(__file__).parent
    output_directory = base_directory / "pdfToText"

    pathlib.Path(output_directory / "result.md").write_text(md_text,"utf-8")



if __name__ == '__main__':
    app.run()
