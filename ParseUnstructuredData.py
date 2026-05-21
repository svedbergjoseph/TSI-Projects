from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import glob

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
print("API KEY FOUND:", os.environ.get("OPENAI_API_KEY") is not None)

client = OpenAI()

class TranscriptExtraction(BaseModel):
    cancelIntent: str
    cancelReasons: list[str]
    saveAttempt: str
    saveOutcome: str
    customerSentiment: str
    urgency: str
    refundRequested: str
    followUpRequired: str
    abstractSummary: str

script_dir = os.path.dirname(os.path.abspath(__file__))
transcript_files = glob.glob(os.path.join(script_dir, "transcript*.txt"))
    
print(f"Found {len(transcript_files)} transcript file(s)...")

for file_path in transcript_files:
    try:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        with open (file_path, "r", encoding = "utf-8") as file:
            transcript_text = file.read()

        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {
                    "role": "system",
                    "content": "You are an expert at structured data extraction. You will be given unstructured text from a transcript between a company agent and a client caller and should convert it into the given structure.",
                },
                {"role": "user", "content": transcript_text},
            ],
            text_format=TranscriptExtraction,
        )

        transcript_extraction = response.output_parsed

        output_name = file_name.replace(".txt", "_output.json")
        output_path = os.path.join(script_dir, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_extraction.dict(), f, indent=2)

        print(f"Report saved to {output_name}")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("All files processed.")