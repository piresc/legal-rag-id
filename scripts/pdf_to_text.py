import os
from PyPDF2 import PdfReader
import pdfplumber
import pytesseract

class PDFConverter:
    def __init__(self, input_dir="indonesian_laws", output_dir="indonesian_laws"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def convert_pdfs(self):
        """Convert all PDFs in input directory to text files"""
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.input_dir, filename)
                text = self._extract_text(pdf_path)
                
                txt_filename = filename.replace(".pdf", ".txt")
                with open(os.path.join(self.output_dir, txt_filename), "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Converted {filename} to text.")
    
    def _extract_text(self, pdf_path):
        """Extract text from PDF using PyPDF2 or fallback to OCR"""
        text = ""
        try:
            # Try extracting text directly
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            
            if not text.strip():
                # Fallback to pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                
                if not text.strip():
                    # Final fallback to OCR
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            text += pytesseract.image_to_string(page.to_image().original) or ""
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        
        return text

if __name__ == "__main__":
    converter = PDFConverter()
    converter.convert_pdfs()