
### **Step 1: Gather Indonesian Legal Documents**
**Goal**: Collect **50–100 legal documents** (laws, regulations, FAQs) in Bahasa Indonesia.  
**Why**: For the MVP, focus on a **narrow niche** (e.g., traffic law, employment disputes) to simplify testing.  

---

#### **Sub-Step 1.1: Identify Free Legal Sources**
Here’s where to scrape/download data (no login/payment required):  
1. **[JDIH Kemenkumham](https://jdihn.go.id/)** (National Law Database):  
   - Use search terms like “UU Lalu Lintas” (traffic law) or “UU Ketenagakerjaan” (employment law).  
   - Filter by format: **PDF** or **DOC**.  

2. **[HukumOnline](https://www.hukumonline.com/)** (Free Section):  
   - Search for “FAQ” or “Tanya Jawab” (Q&A) for common legal questions.  
   - Example: [Traffic Law Q&A](https://www.hukumonline.com/klinik/detail/ulasan/lt5f4a4d0b3f1e3/pidana-lalu-lintas/).  

3. **Mahkamah Agung (Supreme Court) Decisions**:  
   - Use [putusan.mahkamahagung.go.id](http://putusan.mahkamahagung.go.id/) → Search for “putusan” (decisions) with keywords like “pidana lalu lintas”.  

4. **Government Websites**:  
   - Example: [jdih.setkab.go.id](https://jdih.setkab.go.id/) (Cabinet Secretariat’s legal docs).  

---

#### **Sub-Step 1.2: Download Documents**
**For Developers**: Use Python scripts to automate downloads.  

##### **Example Script for JDIHN** (using `requests` and `BeautifulSoup`):  
```python
import requests
from bs4 import BeautifulSoup
import os

# Create folder to save PDFs
os.makedirs("indonesian_laws", exist_ok=True)

# Step 1: Fetch search results for "UU Lalu Lintas"
url = "https://jdihn.go.id/search?categories=Hukum%20Publik&q=UU%20Lalu%20Lintas"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Extract PDF links
pdf_links = []
for link in soup.find_all("a", class_="btn-download"):
    pdf_url = link.get("href")
    if pdf_url.endswith(".pdf"):
        pdf_links.append(pdf_url)

# Step 3: Download PDFs
for i, pdf_url in enumerate(pdf_links[:10]):  # Download first 10 PDFs
    pdf_response = requests.get(pdf_url)
    with open(f"indonesian_laws/law_{i+1}.pdf", "wb") as f:
        f.write(pdf_response.content)
    print(f"Downloaded: law_{i+1}.pdf")
```

##### **Manual Download Shortcut**:  
If coding feels slow, manually download 10 PDFs from [JDIHN](https://jdihn.go.id/) by:  
1. Searching “UU Lalu Lintas”.  
2. Clicking “Download” on each result.  

---

#### **Sub-Step 1.3: Convert PDFs to Text**
Many legal PDFs are text-based, but some are scanned images. Use Python to extract text:  

1. **Install Libraries**:  
   ```bash
   pip install PyPDF2 pdfplumber  # For text extraction
   pip install pytesseract pillow  # For OCR (scanned PDFs)
   ```

2. **Run This Script**:  
```python
import os
from PyPDF2 import PdfReader
import pdfplumber

def pdf_to_text(pdf_path):
    text = ""
    try:
        # Try extracting text directly (for non-scanned PDFs)
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    except:
        # Fallback to OCR for scanned PDFs
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

# Convert all PDFs to text
for filename in os.listdir("indonesian_laws"):
    if filename.endswith(".pdf"):
        text = pdf_to_text(f"indonesian_laws/{filename}")
        with open(f"indonesian_laws/{filename.replace('.pdf', '.txt')}", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Converted {filename} to text.")
```

---

#### **Sub-Step 1.4: Organize Data**
Structure your files like this:  
```
indonesian_laws/  
├── UU_No_22_Tahun_2009_Lalu_Lintas.txt  
├── PP_No_43_Tahun_1993_Kendaraan.txt  
└── ...  
```

---

#### **Sub-Step 1.5: Verify Data Quality**
1. Open a few `.txt` files and check:  
   - Are there garbled characters? → Re-download the PDF.  
   - Does the text include section headers (e.g., “Pasal 362”)?  
2. Remove non-legal content (e.g., watermarks, headers).  

---

### **What’s Next?**  
Once you’ve completed Step 1 (you should have **10–20 text files**), let me know, and I’ll guide you through **Step 2: Preprocessing Data for RAG**.  

---

### **Troubleshooting Tips**  
- **Scanned PDFs Not Extracting?**:  
  Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for your OS, then add this to the script:  
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update path
  ```  
- **Website Blocking Scraping?**:  
  Add a delay between requests:  
  ```python
  import time
  time.sleep(2)  # Wait 2 seconds between downloads
  ```  

---

Let me know when you’ve finished Step 1, and I’ll draft **Step 2** with the same level of detail! 🚀