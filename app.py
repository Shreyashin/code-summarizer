from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import ast
import traceback
import google.generativeai as genai
import requests
from flask import send_from_directory
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
Gmodel = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Hugging Face Spaces API endpoint
HF_API_URL = "https://shin28-code-summarize-api.hf.space/predict"

# Build the prompt dynamically for Gemini
def build_prompt(original_code):
    return f"""
You are an expert Python developer and teacher.
You are provided with an original Python function:
{original_code}

Generate 5 different simplest and smallest counterfactual versions of this function that maintain the same functionality but use different approaches or styles.

IMPORTANT: Your response must be in the exact format shown below:

counterfactuals = {{
    "variant1": '''def function_name(params):
    # implementation
    ''',
    
    "variant2": '''def function_name(params):
    # implementation
    ''',
    
    "variant3": '''def function_name(params):
    # implementation
    ''',
    
    "variant4": '''def function_name(params):
    # implementation
    ''',
    
    "variant5": '''def function_name(params):
    # implementation
    '''
}}
Replace the placeholder code with actual Python code for each variant. 
Do not include any explanations or descriptions outside the code block.
The dictionary should be named 'counterfactuals'.
"""

# Extract counterfactual dictionary from Gemini response
def extract_counterfactual_dict(raw_code):
    # Remove markdown code fences if they exist
    cleaned_code = raw_code.strip()
    if cleaned_code.startswith("```"):
        cleaned_code = "\n".join(line for line in cleaned_code.splitlines() if not line.strip().startswith("```"))
    
    # Parse and extract the dictionary using AST
    tree = ast.parse(cleaned_code)
    
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "counterfactuals":
                    return ast.literal_eval(node.value)
    
    raise ValueError("No 'counterfactuals' dictionary found in the provided code.")

# Get code summary from Hugging Face Spaces API
def get_code_summary(code):
    try:
        response = requests.post(HF_API_URL, json={"code": code})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error calling Hugging Face API: {str(e)}")
        raise


# Dynamically generate counterfactual code variations using Gemini API
def generate_counterfactuals(original_code):
    print(original_code)
    print(type(original_code))
    try:
        # First try using Gemini API
        response = Gmodel.generate_content(build_prompt(original_code))
        print(response)
        counterfactuals = extract_counterfactual_dict(response.text)
        return counterfactuals
    except Exception as e:
        print(f"Error using Gemini API: {str(e)}")
        print("Falling back to default counterfactuals")
        
        # Fallback to hardcoded counterfactuals if Gemini fails
        counterfactuals = {}

        cf1 = re.sub(r'def factorial', 'def compute_factorial', original_code)
        cf1 = re.sub(r'\bfactorial\b', 'compute_factorial', cf1)
        counterfactuals["rename_function"] = cf1

        cf2 = re.sub(r'if n == 0', 'if n <= 1', original_code)
        counterfactuals["change_base_case"] = cf2

        lines = original_code.split('\n')
        for i, line in enumerate(lines):
            if "def " in line:
                indent = ' ' * (len(line) - len(line.lstrip()) + 4)
                lines.insert(i + 1, indent + 'print("Calculating factorial")')
                break
        cf3 = '\n'.join(lines)
        counterfactuals["add_print"] = cf3

        cf4 = """def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result"""
        counterfactuals["iterative"] = cf4

        cf5 = """def factorial(n):
    if n != 0:
        return n * factorial(n-1)
    else:
        return 1"""
        counterfactuals["reverse_condition"] = cf5

        return counterfactuals

# API endpoints
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Get summary from Hugging Face Spaces API
        result = get_code_summary(code)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in summarize: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/counterfactuals', methods=['POST'])
def counterfactuals():
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Generate counterfactuals using Gemini API
        cf_variations = generate_counterfactuals(code)
        
        results = []
        
        for label, cf_code in cf_variations.items():
            # Get summary for each counterfactual from Hugging Face Spaces API
            summary_result = get_code_summary(cf_code)
            
            results.append({
                'label': label,
                'code': cf_code,
                'summary': summary_result['summary'],
                'topWords': summary_result['topWords']
            })
        
        return jsonify({'counterfactuals': results})
    
    except Exception as e:
        print(f"Error in counterfactuals: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # First generate summary for original code
        orig_summary_result = get_code_summary(code)
        
        # Generate counterfactuals
        cf_variations = generate_counterfactuals(code)
        
        # Prepare complete analysis
        results = {
            'original': {
                'code': code,
                'summary': orig_summary_result['summary'],
                'topWords': orig_summary_result['topWords']
            },
            'counterfactuals': []
        }
        
        for label, cf_code in cf_variations.items():
            summary_result = get_code_summary(cf_code)
            
            results['counterfactuals'].append({
                'label': label,
                'code': cf_code,
                'summary': summary_result['summary'],
                'topWords': summary_result['topWords']
            })
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': ['huggingface', 'gemini']})

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
