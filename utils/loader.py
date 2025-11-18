import glob
from langchain.schema import Document

def load_files(folder):
    docs = []

    for f in glob.glob(folder + "/**/*.*", recursive=True):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as file:
                docs.append(Document(
                    page_content=file.read(),
                    metadata={"source": f}
                ))
        except:
            pass

    return docs
