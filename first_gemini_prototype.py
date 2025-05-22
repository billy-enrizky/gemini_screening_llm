import pandas as pd
import google.generativeai as genai
import json
import os
import time
from dotenv  import load_dotenv
load_dotenv()
# --- Configuration ---
# IMPORTANT: Set your API key here or as an environment variable
# Option 1: Set directly in code (less secure for shared scripts)
# API_KEY = "YOUR_GEMINI_API_KEY"
# genai.configure(api_key=API_KEY)

# Option 2: Set as environment variable (more secure)
# and then run: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# For this example, let's assume you've set it as an environment variable
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("🚨 GEMINI_API_KEY environment variable not set.")
    print("Please set it before running the script.")
    exit()
except Exception as e:
    print(f"🚨 Error configuring Gemini API: {e}")
    exit()


# --- Define Criteria (as strings for the prompt) ---
INCLUSION_CRITERIA_LIST = [
    "English language",
    "Effects on RBC properties/function",
    "Liposome-RBC interaction",
    "Relevant study type",
    "Liposome/RBC properties with interaction implications",
    "Applications of liposome-RBC interactions",
    "Theoretical/computational study"
]

EXCLUSION_CRITERIA_LIST = [
    "No liposome-RBC interaction implications",
    "Other cell types without RBC component",
    "Passing mention only",
    "Full text unavailable",  # Note: LLM can only infer this if mentioned in abstract
    "Duplicate publication",  # Note: LLM cannot verify this from a single entry
    "Non-peer-reviewed or preprint" # LLM can try to infer (e.g. from "proceedings", "preprint server")
]

# --- LLM Model Setup ---
# For text-based tasks, gemini-pro is suitable
model = genai.GenerativeModel('models/gemini-2.0-flash')

# --- Helper Function to Create Prompt ---
def create_prompt(title, abstract):
    prompt = f"""
    You are an expert scientific paper screener. Your task is to analyze a given title and abstract and determine if it should be included or excluded based on the provided criteria.

    Here are the criteria:

    Inclusion Criteria:
    {chr(10).join([f"- {c}" for c in INCLUSION_CRITERIA_LIST])}

    Exclusion Criteria:
    {chr(10).join([f"- {c}" for c in EXCLUSION_CRITERIA_LIST])}

    Decision Logic:
    1. First, assess if any Exclusion Criteria are met. If ANY exclusion criterion is met, the decision is "exclude".
    2. If no exclusion criteria are met, then assess Inclusion Criteria.
    3. The paper MUST be in "English language". If not, it's "exclude".
    4. The decision is "include" if ("English language" IS met AND "Effects on RBC properties/function" IS met).
    5. OR the decision is "include" if ("English language" IS met AND "Liposome/RBC properties with interaction implications" IS met).
    6. If none of the above "include" conditions (4 or 5) are met (and no exclusion criteria were met), the decision is "exclude".

    Provide your response as a JSON object with the following keys:
    - "include_criteria_met": A list of strings of ALL satisfied inclusion criteria from the provided list.
    - "exclude_criteria_met": A list of strings of ALL satisfied exclusion criteria from the provided list.
    - "decision": A string, either "include" or "exclude", based on the logic above.
    - "reasoning_summary": A list of strings, detailing your step-by-step thought process. Explain why each listed criterion (include or exclude) was met or not, and how you arrived at the final decision based on the decision logic. Be specific about what parts of the title or abstract led to your conclusions for each criterion.

    Here is the paper:
    Title: "{title}"
    Abstract: "{abstract}"

    Respond ONLY with the JSON object. Do not add any text before or after the JSON.
    """
    return prompt

# --- Function to Process a Single Study Identifier ---
def process_study(study_data):
    if not isinstance(study_data, dict) or 'title' not in study_data or 'abstract' not in study_data:
        return {
            "include_criteria_met": [],
            "exclude_criteria_met": ["Invalid input format"],
            "decision": "exclude",
            "reasoning_summary": ["Input data was not a valid dictionary with 'title' and 'abstract'."]
        }

    title = study_data.get('title', '')
    abstract = study_data.get('abstract', '')

    if not title and not abstract:
         return {
            "include_criteria_met": [],
            "exclude_criteria_met": ["Missing title and abstract"],
            "decision": "exclude",
            "reasoning_summary": ["Both title and abstract are missing."]
        }

    prompt = create_prompt(title, abstract)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                # Safety settings can be adjusted if needed
                # safety_settings=[
                #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                # ],
                generation_config=genai.types.GenerationConfig(
                    # candidate_count=1, # Default
                    # stop_sequences=['...'], # If needed
                    max_output_tokens=20480, # Adjust if JSON is truncated
                    temperature=0.2 # Lower temperature for more deterministic classification
                )
            )
            
            # Debug: print raw response text
            print("--- RAW LLM Response ---")
            print(response.text)
            print("------------------------")

            # The response text should be a JSON string.
            # Sometimes Gemini might add ```json ... ```, so we try to strip that.
            json_text = response.text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            json_text = json_text.strip() # Clean up any leading/trailing whitespace

            result_dict = json.loads(json_text)
            
            # Validate expected keys
            expected_keys = {"include_criteria_met", "exclude_criteria_met", "decision", "reasoning_summary"}
            if not expected_keys.issubset(result_dict.keys()):
                raise ValueError(f"LLM response missing one or more expected keys. Got: {result_dict.keys()}")

            return result_dict

        except json.JSONDecodeError as e:
            print(f"⚠️ JSONDecodeError for title '{title[:50]}...': {e}. Raw response: '{response.text[:200]}...'")
            if attempt < max_retries - 1:
                print(f"Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2**(attempt + 1)) # Exponential backoff
                continue
            else:
                return {
                    "include_criteria_met": [],
                    "exclude_criteria_met": ["LLM output not valid JSON after retries"],
                    "decision": "exclude",
                    "reasoning_summary": [f"LLM did not return valid JSON. Error: {e}. Last raw response: '{response.text[:200]}'"]
                }
        except genai.types.generation_types.BlockedPromptException as e:
             print(f"🚫 Prompt blocked for title '{title[:50]}...': {e}")
             return {
                "include_criteria_met": [],
                "exclude_criteria_met": ["Content blocked by API"],
                "decision": "exclude",
                "reasoning_summary": [f"The prompt or content was blocked by the API due to safety settings. Details: {e}"]
            }
        except Exception as e:
            print(f"🚨 An error occurred while processing title '{title[:50]}...': {e}")
            if attempt < max_retries - 1:
                print(f"Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2**(attempt + 1)) # Exponential backoff
                continue
            else:
                return {
                    "include_criteria_met": [],
                    "exclude_criteria_met": ["LLM API call failed after retries"],
                    "decision": "exclude",
                    "reasoning_summary": [f"An unexpected error occurred with the LLM API call: {e}"]
                }
    return { # Should not be reached if retry logic is correct, but as a fallback
        "include_criteria_met": [],
        "exclude_criteria_met": ["Processing failed after multiple retries"],
        "decision": "exclude",
        "reasoning_summary": ["Failed to get a valid response from the LLM after all retries."]
    }


# --- Main Processing Logic ---
def main():

    my_df = pd.read_pickle("my_df.pkl")
    my_df = my_df.loc[:10,:]

    results_list = []
    for index, row in my_df.iterrows():
        print(f"\nProcessing row {index}...")
        study_info = row['study_identifier']
        
        # Handle cases where 'study_identifier' might be a string representation of a dict
        if isinstance(study_info, str):
            try:
                study_info = json.loads(study_info.replace("'", "\"")) # Basic attempt to fix common string issues
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse 'study_identifier' string in row {index} as JSON. Skipping.")
                results_list.append({
                    "include_criteria_met": [],
                    "exclude_criteria_met": ["Invalid input format for study_identifier"],
                    "decision": "exclude",
                    "reasoning_summary": ["'study_identifier' was a string that could not be parsed as JSON."]
                })
                continue
        
        llm_output = process_study(study_info)
        results_list.append(llm_output)
        
        # Optional: Print intermediate results
        print(f"Title: {study_info.get('title', 'N/A')[:60]}...")
        print(f"Decision: {llm_output.get('decision')}")
        print(f"Include Met: {llm_output.get('include_criteria_met')}")
        print(f"Exclude Met: {llm_output.get('exclude_criteria_met')}")
        print(f"Summary: {llm_output.get('reasoning_summary')}") # Can be verbose
        
        # Be respectful of API rate limits if you have many rows
        time.sleep(1) # Add a small delay if needed

    # Add results to the DataFrame
    # Method 1: As a new column containing the dictionary
    my_df['llm_analysis'] = results_list

    # Method 2: Explode the dictionary into multiple columns (more common)
    for key in ["include_criteria_met", "exclude_criteria_met", "decision", "reasoning_summary"]:
        my_df[f'llm_{key}'] = [r.get(key) for r in results_list]

    print("\n\n--- Final DataFrame with LLM Analysis ---")
    print(my_df.head())

    # You can save it to a CSV or Excel file
    my_df.to_csv("analyzed_studies.csv", index=False)
    # my_df.to_excel("analyzed_studies.xlsx", index=False)

if __name__ == '__main__':
    main()