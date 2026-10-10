"""List the Gemini models your API key can use for text generation.

Run:  python list_models.py
Then pick a 'flash' model and set it:  $env:GEMINI_MODEL="the-name-you-chose"
"""
import os

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name.replace("models/", ""))
