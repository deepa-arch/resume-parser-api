#!/usr/bin/env python
# app.py - Flask Resume Parser API

import os
import re
import json
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text as pdf_extract
from groq import Groq
import subprocess

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Get API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

def convert_to_pdf(doc_path):
    """Convert DOC/DOCX to PDF using LibreOffice"""
    pdf_path = doc_path.rsplit(".", 1)[0] + ".pdf"
    try:
        if os.path.exists(pdf_path):
            return pdf_path

        result = subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", doc_path,
            "--outdir", os.path.dirname(doc_path)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

        if os.path.exists(pdf_path):
            return pdf_path
    except Exception as e:
        logging.error(f"Error converting to PDF: {e}")
    return None

def extract_text_from_file(file_path):
    """Extract text from PDF, DOC, DOCX, or TXT files"""
    ext = file_path.rsplit(".", 1)[1].lower()
    
    try:
        if ext == "pdf":
            text = pdf_extract(file_path)
            if text.strip():
                return clean_text(text)
            return ""

        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read().strip())

        elif ext in ["doc", "docx"]:
            pdf_path = convert_to_pdf(file_path)
            if pdf_path:
                text = pdf_extract(pdf_path)
                return clean_text(text)
            return ""
            
    except Exception as e:
        logging.error(f"Text extraction error: {e}")
    
    return ""

def parse_resume_with_ai(resume_text):
    """Parse resume using Groq AI"""
    
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}
    
    prompt = f"""You are an expert resume parser. Extract ALL information from the following resume and return it as a structured JSON object.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON, no markdown, no explanation
- Use null for missing fields
- Extract everything you find

Resume Text:
{resume_text}

Return this EXACT JSON structure:
{{
  "personal_info": {{
    "first_name": "string or null",
    "last_name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "location": "string or null",
    "linkedin": "string or null",
    "github": "string or null",
    "portfolio": "string or null",
    "summary": "string or null"
  }},
  "education": [
    {{
      "degree": "string or null",
      "field_of_study": "string or null",
      "institution": "string or null",
      "location": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "grade": "string or null"
    }}
  ],
  "experience": [
    {{
      "job_title": "string or null",
      "company": "string or null",
      "location": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "responsibilities": ["string"],
      "achievements": ["string"],
      "technologies": ["string"]
    }}
  ],
  "projects": [
    {{
      "name": "string or null",
      "description": "string or null",
      "role": "string or null",
      "technologies": ["string"],
      "link": "string or null"
    }}
  ],
  "skills": {{
    "technical_skills": ["string"],
    "programming_languages": ["string"],
    "frameworks": ["string"],
    "tools": ["string"],
    "databases": ["string"],
    "soft_skills": ["string"]
  }},
  "certifications": [
    {{
      "name": "string or null",
      "issuer": "string or null",
      "date": "string or null",
      "credential_id": "string or null"
    }}
  ],
  "hackathons": [
    {{
      "name": "string or null",
      "organizer": "string or null",
      "date": "string or null",
      "achievement": "string or null",
      "project": "string or null"
    }}
  ],
  "awards": [
    {{
      "name": "string or null",
      "issuer": "string or null",
      "date": "string or null",
      "description": "string or null"
    }}
  ]
}}"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert resume parser. Always return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        
        # Clean response
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = content.strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        parsed = json.loads(content)
        return parsed
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {"error": "Failed to parse JSON response", "details": str(e)}
    except Exception as e:
        logging.error(f"API error: {e}")
        return {"error": str(e)}

@app.route("/", methods=["GET"])
def index():
    """API documentation endpoint"""
    return jsonify({
        "service": "Resume Parser API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/parse": {
                "method": "POST",
                "description": "Upload and parse resume",
                "content_type": "multipart/form-data",
                "parameters": {
                    "file": "Resume file (PDF, TXT, DOC, DOCX)"
                },
                "max_file_size": "16MB"
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        },
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "groq_api_configured": bool(GROQ_API_KEY),
        "model": MODEL_NAME
    }), 200

@app.route("/parse", methods=["POST"])
def parse_resume():
    """Parse uploaded resume"""
    try:
        # Check if file is present
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type",
                "allowed_types": list(ALLOWED_EXTENSIONS)
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        logging.info(f"File saved: {file_path}")

        # Extract text
        resume_text = extract_text_from_file(file_path)
        
        if not resume_text:
            return jsonify({"error": "Failed to extract text from resume"}), 500

        logging.info(f"Extracted {len(resume_text)} characters")

        # Parse with AI
        parsed_data = parse_resume_with_ai(resume_text)

        # Clean up file
        try:
            os.remove(file_path)
            if file_path.endswith(('.doc', '.docx')):
                pdf_path = file_path.rsplit(".", 1)[0] + ".pdf"
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
        except Exception as e:
            logging.warning(f"Failed to clean up files: {e}")

        # Return result
        return jsonify({
            "status": "success",
            "data": parsed_data,
            "metadata": {
                "filename": file.filename,
                "text_length": len(resume_text)
            }
        }), 200

    except Exception as e:
        logging.error(f"Error processing resume: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)