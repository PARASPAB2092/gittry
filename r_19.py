import streamlit as st
import os
import zipfile
import glob
import shutil
import re
import csv
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import chunking_aashika
from pydantic import BaseModel, SecretStr
from typing import Any

# Updated class for handling SecretStr and Pydantic v2 compatibility
class MyModel(BaseModel):
    secret: SecretStr

    def __get_pydantic_json_schema__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().__get_pydantic_json_schema__(*args, **kwargs)
        if 'properties' in schema:
            schema['properties']['secret']['type'] = 'string'
            schema['properties']['secret']['description'] = 'This is a secret string field.'
        return schema

# Function to extract function names from an R script
def extract_functions_from_r_script(file_path):
    functions = []
    try:
        with open(file_path, "r") as file:
            content = file.read()
            # Match patterns for R function definitions
            matches = re.findall(r"(\w+)\s*(<-|=)\s*function", content)
            functions.extend(match[0] for match in matches)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return functions

# Function to map functions to their respective scripts and save to CSV
def search_and_extract_functions(directory, output_csv):
    data = []
    with open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Function Name", "Script Name"])  # CSV header

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".R"):
                    file_path = os.path.join(root, file)
                    functions = extract_functions_from_r_script(file_path)
                    for function in functions:
                        csv_writer.writerow([function, file])
                        data.append([function, file_path])
    functions_df = pd.DataFrame(data, columns=["Function Name", "Script Name"])
    return functions_df

def concat_files_in_order(input_folder, output_file, base_filename):
    file_pattern = os.path.join(input_folder, f"{base_filename}.R_chunk_*.py")
    print('#####################', output_file)

    files = sorted(glob.glob(file_pattern), key=os.path.getmtime)

    with open(output_file, 'w') as output:
        for file in files:
            with open(file, 'r') as f:
                output.write(f.read())
                output.write('\n')

def network_graph_df(r_script_path, df, network_df):

    with open(r_script_path, "r") as file:
        r_script_content = file.read()
    
    # Find all function calls in the R script using regex
    function_calls = re.findall(r"\b(\w+)\s*\(", r_script_content)
    function_calls = set(function_calls)

    for func in function_calls:
        if func in df['Function Name'].values:
            # Get the file where the function is defined
            try:
                file_defined = df.loc[df['Function Name'] == func, 'Script Name'].iloc[0]
                if r_script_path != os.path.join("extracted_files", file_defined):
                    #dependencies.append(f"from {file_defined} import {func}")
                    network_df.loc[len(network_df)] = [func, file_defined, r_script_path]
            except Exception as e:
                print(f"Error processing function {func}: {e}")
                continue

def network_graph(df):
    # Create a directed graph
    G = nx.DiGraph()
    # need to write code to generate network graph

def reverse_engineer(r_file_path):
    with open(r_file_path, 'r') as r_code:
        r_script_content1 = r_code.read()
        r_script_content = f" {{ {r_script_content1} }}"
        r_script_content = r_script_content1.replace("{", "{{").replace("}", "}}")
        print(';;;;;;;;;;;;;;;', r_script_content)

        system_message = "You are a helpful assistant who can understand code written in R and SparkR.."

        human_message = f"""
        I need you to break down all the transformations, functions, variables and reverse-engineer the following R or SparkR script.
        Provide a detailed explanation of what the script does, including the purpose of each section, the functions being called, all the if-else conditions, 
        the data structures involved, name of the dataframe and its columns, the computation logic, and the overall functionality.
        Note down the transformations for each dataframe carefully along with column names.
        For training One-Class SVM Model, all the parameters along with their values should be noted.
        Do not write any Python code yet, just focus on fully understanding the R code and explaining its functionality step by step: {r_script_content}
        Clearly SPECIFY the DATASET names used and its columns involved in SVM fit and prediction.
        Mention clearly which step is in R and which is in SparkR.
        """

        prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])
        chat_model = ChatGroq(groq_api_key="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", model_name="llama3-8b-8192", temperature=0.7)
        response = prompt | chat_model
        output = response.invoke({"r_code": human_message})
        reverse_code1 = output.content
        reverse_code = f" {{ {reverse_code1} }}"
        reverse_code = reverse_code1.replace("{", "{{").replace("}", "}}")

        with open('reverse_code_12dec.txt', "a") as python_file:
            python_file.write("\n\n# Python equivalent of the provided R code:\n\n")
            python_file.write(output.content)
            #st.write('reverse code generated')
    return reverse_code


def generate_python(reverse_code, output_directory, r_file_path):
    system_message = "You are a python/pyspark code developer who can write a syntactically correct python code using the description. Ensure Python code adheres to PEP 8 standards."

    human_message = f"""
    Now that we have the functionality required in the reverse code, convert the logic, data structures, and functions into Python code.
    Keep the structure and functionality the same while adapting it to Python syntax and libraries.
    Ensure that the equivalent Python code maintains the same outputs, column names of dataframe, and behavior as the R script.
    If the R script uses specific libraries or methods, suggest the appropriate Python equivalents (e.g., pandas for data frames, matplotlib for plotting, pyspark for sparkR etc.).
    If the R scripts references other scripts to import functions, the python script should do AS IS.
    While calling the functions, print the return dataframe.
    Read the reverse code carefully without missing any line during conversion. True and False should be in capitals.
    Use a copy of the dataframe when passing to a function as a parameter.
    The filenames/package names are case-sensitive.
    DO NOT change the column names in the dataframes.
    \n{reverse_code}"""

    prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])
    chat_model = ChatGroq(groq_api_key="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", model_name="mixtral-8x7b-32768", temperature=0.4)
    response = prompt | chat_model
    output1 = response.invoke({"text": human_message})
    print('....', output1.content)

    python_file_name = os.path.splitext(os.path.basename(r_file_path))[0] + ".py"
    python_file_path = os.path.join(output_directory, python_file_name)

    code_block = re.search(r'```python(.*?)```', output1.content, re.DOTALL)
    if code_block:
        python_code = code_block.group(1).strip()
    else:
        python_code = output1.content

    with open(python_file_path, "w") as python_file:
        python_file.write("# Python equivalent of the provided R code:\n\n")
        python_file.write(python_code)


def validate_python(python_file_path, r_file_path, output_directory, base_filename):
    with open(python_file_path, 'r') as file:
        content1 = file.read()
    content = f" {{ {content1} }}"
    content = content1.replace("{", "{{").replace("}", "}}")
    system_message = "You are a python/pyspark code developer who needs to validate if the python code is correct or not syntactically."
    human_message = f"""
    If the python code is not correct or has any unused commands, correct it to create a syntactically correct python code.
    DO NOT change the logic.
    \n{content}"""

    prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])
    chat_model = ChatGroq(groq_api_key="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", model_name="mixtral-8x7b-32768", temperature=0.4)
    response = prompt | chat_model
    output1 = response.invoke({"text": human_message})
    print('....', output1.content)

    python_file_name = os.path.splitext(os.path.basename(r_file_path))[0] + ".py"
    python_file_path = os.path.join(output_directory, python_file_name)

    code_block = re.search(r'```python(.*?)```', output1.content, re.DOTALL)
    print('////', code_block)

    if code_block:
        python_code = code_block.group(1).strip()
    else:
        python_code = output1.content

    python_file_path = os.path.join(output_directory, base_filename) + '.py'

    with open(python_file_path, "w") as python_file:
        python_file.write("# Python equivalent of the provided R code:\n\n")
        python_file.write(python_code)


# Streamlit App
def main():

    st.title('R to Python conversion')

    uploaded_zip = st.file_uploader("Upload a zip file containing the R scripts", type="zip")
    if uploaded_zip is not None:
        extract_file_path = 'extracted_files'
        r_file_path = 'chunks'
        output_directory = 'converted_py_scripts_chunks'
        converted_directory = "converted_py_scripts"
        network_graph_csv, output_csv = "network_graph.csv", "functions_in_r_scripts.csv"
        os.makedirs(extract_file_path, exist_ok=True)
        os.makedirs(r_file_path, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(converted_directory, exist_ok=True)
        network_df = pd.DataFrame(columns=["Function Name", "Source", "Destination"])

        # Cleanup previous files
        for file in glob.glob(os.path.join(extract_file_path, '*')):
            os.remove(file)
        if os.path.exists('reverse_code_12dec.txt'):
            os.remove('reverse_code_12dec.txt')
        for file in glob.glob(os.path.join(r_file_path, '*')):
            os.remove(file)
        for file in glob.glob(os.path.join(output_directory, '*.py')):
            os.remove(file)
        for file in glob.glob(os.path.join(converted_directory, '*.py')):
            os.remove(file)

        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_file_path)
            st.success(f'Files extracted to {extract_file_path}')

        #Extract functions
        function_df=search_and_extract_functions(extract_file_path, output_csv)
        ## Implementing chunking & network graph
        for root, dirs, files in os.walk(extract_file_path):
            for file in files:
                extract_file_path1 = os.path.join(root, file)
                network_graph_df(extract_file_path1,function_df, network_df)
                chunking_aashika.main_chunk(extract_file_path1, r_file_path)
        network_df.to_csv(network_graph_csv, index=False)
        st.success(f'network graph csv created at {network_graph_csv}')
        #network_graph(network_df)
        # need to add code to use network graph to pass dependency in the prompt

        # Process each R file and generate Python code
        for root, dirs, files in os.walk(r_file_path):
            for file in files:
                r_file_path = os.path.join(root, file)
                base_filename = os.path.basename(r_file_path).split("_")[0].rsplit(".", 1)[0]
                reverse_code = reverse_engineer(r_file_path)
                generate_python(reverse_code, output_directory, r_file_path)
        
        st.write('reverse code generated')
        # Concatenate all Python files and validate them
        for root, dirs, files in os.walk(extract_file_path):            
            for file in files:
                base_filename = os.path.basename(file).split(".R")[0]
                python_file_path = os.path.join(output_directory, base_filename) + '_temp.py'
                concat_files_in_order(output_directory, python_file_path, base_filename)
                validate_python(python_file_path, r_file_path, converted_directory, base_filename)
        st.success(f'All files processed and saved at the path: {converted_directory}')

if __name__ == "__main__":
    main()
