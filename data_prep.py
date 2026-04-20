import os
import re
import fitz
import json
import unicodedata
from langchain_core.documents import Document

def extract_pdf(
    path: str,
    output: str
):
    text_li = []
    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text_li.append(f"\n--- Page {page_num} ---\n{text}")

    total = "\n".join(text_li)
    with open(output, "w", encoding = "utf-8") as f:
        f.write(total)
    return

def pdf_to_txt(
    targets: list = ["book", "sheet"]
):
    for target in targets:
        tar_li = os.listdir(f"./sources/pdf/{target}")
        outpath = f"./sources/txt/{target}"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for file in tar_li:
            extract_pdf(
                path = f"./sources/pdf/{target}/{file}",
                output=f"{outpath}/{file.replace('.pdf', '.txt')}"
            )

def data_cleaning(file: str):
    file = unicodedata.normalize("NFKC", file)
    file = file.replace("\xa0", " ")
    file = re.sub(r"[ \t]+", " ", file)
    return file.strip()

def text_to_sec(
    filepath: str,
    filename: str = ""
):
    # extract section
    with open("./sources/table_of_content.json", "r", encoding = "utf-8") as f:
        sec_li = json.load(f)[filename]

    with open(f"{filepath}/{filename}.txt", "r", encoding = "utf-8") as f:
        file = f.read()
    file = data_cleaning(file)
    file_li = file.splitlines()
    idx_li = []
    for idx, content in enumerate(file_li):
        if re.match(r"--- page \d+ ---", content.lower()):
            num = int(re.findall(r"\d+", content.lower())[0])
            if num in sec_li:
                idx_li.append(idx)
            elif num > sec_li[-1]:
                break
    
    sec_li = file_li[idx_li[0] : idx_li[-1]]
    title = {}
    for line in sec_li:
        if any(s.isalpha() for s in line) and any(s.isdigit() for s in line) and "page" not in line.lower():
            m = re.match(r"^(?P<title>.+?)\s+(?P<page>\d+)$", line.strip())
            if m:
                title[m.group("page")] = " ".join(m.group("title").split())
            # title[line.split()[-1]] = " ".join(line.split()[:-1])
    structured = []
    index_li = [0]
    for idx, content in enumerate(file_li):
        if re.match(r"--- page \d+ ---", content.lower()):
            num = re.findall(r"\d+", content)[0]
            if num in title.keys():
                structured.append(
                    Document(
                        page_content = "\n".join(file_li[index_li[-1]: idx]),
                        metadata = {
                            "bookname": filename,
                            "section": title[num]
                        }
                    )
                )
                index_li.append(idx)

    structured.append(
        Document(
            page_content = "\n".join(file_li[index_li[-1]:]),
            metadata = {
                "bookname": filename,
                "section": title[[i for i in title.keys()][-1]]
            }
        )
    )
    return structured
                
def load_doc():
    all_docs = []
    book_li = os.listdir("./sources/txt/book")
    sheet_li = os.listdir("./sources/txt/sheet")
    for file in book_li:
        with open(f"./sources/txt/book/{file}", "r", encoding = "utf-8") as f:
            text = f.read()
        docs = text_to_sec(
            filepath = "./sources/txt/book/",
            filename = file.replace(".txt", "")
        )
        all_docs.extend(docs)
    
    for file in sheet_li:
        with open(f"./sources/txt/sheet/{file}", "r", encoding = "utf-8") as f:
            text = f.read()
        text = data_cleaning(file=text)
        all_docs.append(
            Document(
                page_content = text,
                metadata = {
                    "bookname": file.replace(".txt", ""),
                    "section": ""
                }
            )
        )
    
    return all_docs
    

if __name__ == "__main__":

    # pdf_to_txt()
    load_doc()
    
    