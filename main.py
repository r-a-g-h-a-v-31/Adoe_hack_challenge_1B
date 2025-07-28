import os
import json
import pdfplumber
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("model/")  # multilingual model
def load_challenge_input(path):
    #to read challenge1b_input.json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        persona = data["persona"]["role"]
        task = data["job_to_be_done"]["task"]
        filenames = [doc["filename"] for doc in data["documents"]]
        return persona, task, filenames

def extract_sections_from_pdf(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=0):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": text.strip(),
                    "section_title": text.strip().split('\n')[0][:60]
                })
    return sections

def rank_sections(sections, query, top_k=5):
    texts = [s["text"] for s in sections]
    embeddings = model.encode(texts, convert_to_tensor=True)
    query_embedding = model.encode([query], convert_to_tensor=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    for i, score in enumerate(scores):
        sections[i]["similarity"] = float(score)

    # Group by document, pick top section per document
    doc_to_top_section = {}
    for section in sorted(sections, key=lambda x: -x["similarity"]):
        doc = section["document"]
        if doc not in doc_to_top_section:
            doc_to_top_section[doc] = section
        if len(doc_to_top_section) == top_k:
            break

    #Return top_k diverse sections sorted by importance
    top_sections = list(doc_to_top_section.values())
    return sorted(top_sections, key=lambda x: -x["similarity"])

def save_results_to_json(ranked_sections, persona, task, output_path):
    metadata = {
        "input_documents": list({s["document"] for s in ranked_sections}),
        "persona": persona,
        "job_to_be_done": task,
        "processing_timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    }

    top_sections = ranked_sections[:5]

    output = {
        "metadata": metadata,
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, sec in enumerate(top_sections, start=1):
        output["extracted_sections"].append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": rank,
            "page_number": sec["page"]            
        })
        output["subsection_analysis"].append({
            "document": sec["document"],
            "refined_text": sec["text"],
            "page_number": sec["page"]            
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")

def main():
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # load persona, task, and filenames from challenge1b_input.json
    challenge_input_path = os.path.join(input_dir, "challenge1b_input.json")
    persona, task, pdf_filenames = load_challenge_input(challenge_input_path)
    query = f"{persona} needs to: {task}"

    all_sections = []

    # iterate over the provided filenames
    for filename in pdf_filenames:
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            print(f"Reading {filename}...")
            sections = extract_sections_from_pdf(filepath)
            all_sections.extend(sections)
        else:
            print(f"File not found: {filename}")

    if not all_sections:
        print("No valid sections found in any PDF.")
        return

    ranked = rank_sections(all_sections, query)

    output_file = os.path.join(output_dir, "persona_output.json")
    save_results_to_json(
        ranked_sections=ranked,
        persona=persona,
        task=task,
        output_path=output_file
    )

if __name__ == "__main__":
    main()
