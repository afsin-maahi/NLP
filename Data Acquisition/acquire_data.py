# ============================================================
# DATA ACQUISITION FOR NLP
# Demonstrates how to acquire textual data from different sources
# ============================================================

from pathlib import Path
import json
import requests
import pandas as pd
import fitz
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from docx import Document
from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


print("\n========== Local Text File ==========\n")
with open(DATA_DIR / "dataset.txt", "r", encoding="utf-8") as file:
    data = file.read()
print(data)


print("\n========== Multiple Text Files ==========\n")
all_text = ""
for file in DATA_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        all_text += f.read() + "\n"
print(all_text)


print("\n========== PDF ==========\n")
doc = fitz.open(DATA_DIR / "Template.pdf")
print("Pages:", doc.page_count)
page = doc[0]
print(page.get_text())


print("\n========== CSV ==========\n")
df = pd.read_csv(DATA_DIR / "questions.csv")
print(df.head())


print("\n========== JSON ==========\n")
with open(DATA_DIR / "dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)
for item in data:
    print(item["question"])
    print(item["answer"])
    print()


print("\n========== Excel ==========\n")
df = pd.read_excel(DATA_DIR / "employees.xlsx")
print(df.head())


print("\n========== DOCX ==========\n")
doc = Document(DATA_DIR / "sample.docx")
for para in doc.paragraphs:
    print(para.text)


print("\n========== XML ==========\n")
tree = ET.parse(DATA_DIR / "sample.xml")
root = tree.getroot()
for child in root:
    print(child.tag, child.text)


print("\n========== Web Scraping ==========\n")
url = "https://example.com"
headers = {
    "User-Agent": "Mozilla/5.0"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
print(soup.get_text())


print("\n========== REST API ==========\n")
url = "https://jsonplaceholder.typicode.com/posts"
response = requests.get(url)
data = response.json()
print(data[:2])


print("\n========== Kaggle Dataset ==========\n")
df = pd.read_csv(DATA_DIR / "train.csv")
print(df.head())


print("\n========== Multiple PDFs ==========\n")
pdf_folder = DATA_DIR / "pdfs"
all_text = ""
for pdf_file in pdf_folder.glob("*.pdf"):
    pdf = fitz.open(pdf_file)
    for page in pdf:
        all_text += page.get_text()
print(all_text)


print("\n========== Hugging Face Dataset ==========\n")
dataset = load_dataset("imdb")
print(dataset["train"][0])