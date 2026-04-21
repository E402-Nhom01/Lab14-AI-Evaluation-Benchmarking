from typing import Union, List
import pdftotext


class PDFReader:
    def __init__(self, pdf_paths: Union[List[str], str]):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        self.pdf_paths = pdf_paths

    def read(self) -> list[str]:
        texts = []

        for pdf_path in self.pdf_paths:
            with open(pdf_path, "rb") as file:
                pdf = pdftotext.PDF(file)
                texts.append("\n\n".join(pdf))

        return texts