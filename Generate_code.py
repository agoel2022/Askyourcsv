import json
import os
import shutil
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import base64
# import requests

#import openai

# from huggingface_hub import Repository
# from text_generation import Client

#from sentence_transformers import SentenceTransformer, util

# import requests
import re
import json
import time

import pandas as pd

# def filter_phi_columns(columns):
#     # Replace this logic with your own criteria for identifying PHI columns.
#     # For example, you can check for column names that suggest PHI data.

#     safe_harbor_list = ['Names', 'Geography','location','street address', 'city', 'county', 'zip code', 'Dates', 'birthdate', 'admission date',
#                         'discharge date', 'date of death', 'Telephone ', 'Fax numbers', 'Email addresses',
#                         'Social Security ','SSN', 'Medical record ', 'Health plan beneficiary',
#                         'Account', 'Certificate license ', 'Vehicle identifiers',
#                         'serial' 'license plate ', 'Device identifiers', 'Web URLs', 'IP addresses',
#                         'Biometric identifiers', 'fingerprints and voice', 'Full face photos',
#                         'unique identifying ', 'codes']
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#     phi_list = set()
#     for col in columns:
#         for pii_col in safe_harbor_list:
#             embedding_1 = model.encode(col, convert_to_tensor=True)
#             embedding_2 = model.encode(pii_col, convert_to_tensor=True)
#             compare_value = util.pytorch_cos_sim(embedding_1, embedding_2).numpy()[0][0]
#             if compare_value > 0.45:
#                 phi_list.add(col)
#                 print(col, pii_col,compare_value)
#     phi_list = list(phi_list)
#     if len(phi_list)>0:
#         return phi_list,True
#     else:
#         return [],False

global result
# Define a function to simulate user authentication
def authenticate(username, password):
    # You can implement your own authentication logic here.
    # For simplicity, we'll use hardcoded credentials.
    valid_username = "admin"
    valid_password = "Welcome"

    if username == valid_username and password == valid_password:
        return True
    else:
        return False

class generate_code():


  url=''
  #api_key='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'
  #api_key='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'
  #openai.api_key=api_key

  def __init__(self,df):
    self.df=df
    self.num_rows=len(df)
    self.num_columns=len(df.columns)
    self.data_types=df.dtypes
    self.default_import = "import pandas as pd"
    self.global_var="global result"
    self.save_charts_path=os.getcwd()

  # def query(self,payload):
  #   response=openai.Completion.create(
  #     engine="text-davinci-002",
  #     prompt=payload["inputs"],
  #     temperature=payload["temperature"],
  #     max_tokens=payload["max_new_tokens"])

  #   return response['choices'][0]['text'].strip()

  def query(self,payload):
    # response=requests.post(self.url,json=payload)
    # return response.json()['response']
    code='''
# TODO import all the dependencies required
import pandas as pd
import matplotlib.pyplot as plt

def analyze_data(df: pd.DataFrame) -> dict:
    """
    Analyze the data
    1. Prepare: Preprocessing and cleaning data if necessary
    2. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)
    3. Analyze: Conducting the actual analysis (if the user asks to plot a chart, save it to an image in /content/temp_chart.png and do not show the chart.)
    4. Output: return a dictionary of:
        - type (possible values "text", "number", "dataframe", "plot")
        - value (can be a string, a dataframe, or the path of the plot, NOT a dictionary)
    Example output: { "type": "text", "value": "The average loan amount is $15,000." }
    """

    # Prepare: No preprocessing or cleaning required in this case

    # Process: Group the data by ICD10code and calculate the total claims for each code
    icd10code_total_claims = df.groupby('icd10code')['claim_amount'].sum()

    
    output = {
        "type": "dataframe",
        "value": icd10code_total_claims
    }

    return output

'''
    return code


  def prompt_generation(self,question):
    columns=self.df.columns
    col_list=','.join(columns)
    prompt=f'''Write a python code on pandas dataframe df with columns:{col_list}.
    The pandas dataframe already exists with the following schema:
    {self.df.dtypes}
    The code assumes the dataframe df while the below code.
    The code should be able to display results for the following user query:
    {question}
    The code should also print the final results using the print statement
    'Do not add comments in the code'
    The code must be executed when passed to exec function
    'Always assume that dataframe df already exists'
    'The code should import the necessary modules'
    'Try to avoid using loops while writing the code if possible'
'''
    return prompt

  def prompt_generation2(self,question):
    columns=self.df.columns
    col_list=','.join(columns)
    #meta_data=self.df.dtypes
    prompt=f'''You are provided with the following pandas DataFrames:df
    The DataFrame df contains columns:{col_list}.
    The metadata of the dataframe df is :
    {self.df.dtypes}

    <conversation>
    {question}
    </conversation>



    'Try to avoid using loops while writing the code if possible'
    'Always assume that dataframe df already exists'
    'The code should import the necessary modules'

    This is the initial python code to be updated:
```python
# TODO import all the dependencies required
{self.default_import}

def analyze_data(df:pd.DataFrame) -> dict:
    \"\"\"
    Analyze the data
    1. Prepare: Preprocessing and cleaning data if necessary
    2. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)
    3. Analyze: Conducting the actual analysis (if the user asks to plot a chart save it to an image in {self.save_charts_path}/temp_chart.png and do not show the chart.)
    4. Output: return a dictionary of:
    - type (possible values "text", "number", "dataframe", "plot")
    - value (can be a string, a dataframe or the path of the plot, NOT a dictionary)
    Example output: {{ "type": "text", "value": "The average loan amount is $15,000." }}
    \"\"\"
```

Using the provided dataframes (`df`), update the python code based on the last question in the conversation.
{question}

Updated code:

'''
    return prompt

  def correcterrorprompt(self,question,code,error_returned):
    columns=self.df.columns
    col_list=','.join(columns)
    #meta_data=self.df.dtypes
    prompt=f'''You are provided with the following pandas DataFrames:df
    The DataFrame df contains columns:{col_list}.
    The metadata of the dataframe df is :
    {self.df.dtypes}

    The user asked the following question:
    {question}

    You generated this python code:
    {code}

    It fails with the following error:
   {error_returned}

   Correct the python code and return a new python code that fixes the above mentioned error.
   'Do not generate the same code again'
   'Try to avoid using loops while writing the code if possible'
    'Always assume that dataframe df already exists'
    'The code should import the necessary modules'


Updated code:

'''
    return prompt


  @staticmethod
  def clean_code(code):
    read_patt=re.compile(r'pd\.read\_csv\([A-Za-z0-9]*\.csv\)')
    code=re.sub(read_patt,'',code)
    matplotlib_patt=re.compile(r"\%matplotlib inline")
    code=re.sub(matplotlib_patt,'',code)
    return code


  def generate(self,
            original_prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0
        ):

            prompt=original_prompt
            generate_kwargs = {
                "temperature":temperature,
                "max_new_tokens":max_new_tokens,
                "top_p":top_p,
                "repetition_penalty":repetition_penalty,
                "do_sample":True,
                "seed":42
            }

            suffix=''

            prompt_dict={"inputs":prompt,**generate_kwargs}
            code_response=self.query(prompt_dict)

            # Check in the code that analyze_data function is called.
            # If not, add it.



            # #for i in range(5):
            # prompt_dict={"inputs":prompt,**generate_kwargs}
            # code_response=self.query(prompt_dict)
            # prompt=code_response

            match = re.search(
              "(```python)(.*)(```)",
               code_response.replace(prompt+suffix, ""),
                re.DOTALL | re.MULTILINE,
                            )
            # if match:
            #   break


            final_code=code_response.replace(original_prompt+suffix, "")
            if " = analyze_data(" not in code_response:
                final_code += "\n\nresult = analyze_data(df)"
            final_code=self.global_var+"\n\n"+final_code
            print("final code:\n"+final_code)

            #final_code=final_code.split("\'''")
            #final_code=final_code.split("You can use the below code to get the answer:")

            return final_code
  @staticmethod
  def execute_code(df,code):
   global result
   try:
    #   print("code:\n"+code)
    #   print("-----------------")
      exec(code,globals(),locals())
      return result
   except Exception as exc:
      return exc

def ask_csv(df,question,show_code=False):
  code_generator=generate_code(df)
  prompt=code_generator.prompt_generation2(question=question)
  code=code_generator.generate(prompt)
#   print("code returned by generate "+code)
  code=code_generator.clean_code(code)
#   print("-----------------------------------")
#   print("code returned by clean code "+code)
  try:
      result=code_generator.execute_code(df,code)
      if show_code:
        print(code)
      else:
          pass
      return result

  except Exception as exc:
    error_caught=code_generator.execute_code(df,code)
    prompt_updated=code_generator.correcterrorprompt(question,code,error_caught)
    code_updated=code_generator.generate(prompt_updated)
    code_updated=code_generator.clean_code(code_updated)
    try:
        result=code_generator.execute_code(df,code_updated)
        if show_code:
          print(code_updated)
        else:
          pass
        return result
    except:
         error_caught_again=code_generator.execute_code(df,code_updated)
         if show_code:
          print(code_updated)
         else:
          pass
         return "The code failed with the following exception: "+error_caught_again

def identify_data_types(result_df):
    numeric_cols=result_df.select_dtypes(include=[np.int64,np.int, np.float64,np.number]).columns.tolist()
    categorical_cols=[col for col in result_df.columns if col not in numeric_cols]
    
    return numeric_cols, categorical_cols

def generate_column_config(result_df):
    column_config = {}
    try:
        numeric_cols,categorical_cols=identify_data_types(result_df)

        for column_name in numeric_cols:
            # Define configuration options for each column here
            column_config[column_name] =st.column_config.ProgressColumn(column_name,
                                                        min_value=result_df[column_name].min(),
                                                        max_value=result_df[column_name] .max(),
                                                        format="%d")
        # Add more configuration options as needed

        # for column_name in categorical_cols: #Make all categorical columns as bold
        #         column_config['fontWeight'] = 'bold'
            return column_config
    except Exception as exc:
        return 1
    
    


def generate_charts(result_df):
    try:
        numeric_cols, categorical_cols = identify_data_types(result_df)
        for num_column in numeric_cols:
            st.subheader(f'Distribution of {num_column}')
            fig, ax = plt.subplots()
            sns.distplot(result_df[num_column], ax=ax)
            st.pyplot(fig)

            if len(categorical_cols)>0:
                for cat_column in categorical_cols:
                    chart_df=result_df.groupby(cat_column)[num_column].sum().reset_index()
                    # Generate bar plot for numeric columns
                    st.subheader(f'Distribution of {num_column} with {cat_column}')
                    #fig, ax = plt.subplots()
                    st.bar_chart(y=cat_column, x=num_column, data=chart_df)
                    #st.pyplot(fig)
    except Exception as exc:
        pass

    

# Streamlit app

st.set_page_config(layout="wide")
st.title("Login Page")

with open("Exl_logo_rgb_orange_pos.jpg", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:20%;">
            <img src="data:image/png;base64,{data}" width="100" height="150">
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("EXL Services")
    st.sidebar.markdown(''':orange[We make sense of data and drive your business forward]''')
    st.sidebar.header("About Us")
    about_us='''
    At EXL, it’s all about outcomes—your outcomes—and delivering success on your terms. 
    Share your goals with us and together, 
    we’ll optimize how you leverage data to drive your business forward.
 '''
    st.sidebar.markdown(about_us)
    # Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

    # Display login section
st.subheader("Login")
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if authenticate(username, password):
        st.session_state.logged_in = True
        st.success("Login successful!")
        welcome_string=f"Welcome, {username} :white_check_mark:"
        st.markdown(welcome_string)
        time.sleep(2)
    else:

        st.error("Login failed. Please try again.")

    # Check if the user is logged in before displaying upload and text boxes
if st.session_state.logged_in:
        # Display file upload section below login
    st.subheader("Upload CSV or Excel File")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
            # Read and display the uploaded file
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine='openpyxl')


        # filter_condition=filter_phi_columns(list(df.columns))
        # if filter_condition[1]:
        #     list_data=",".join(filter_condition[0])
        #     st.error("Datset uploaded contain PHI columns - "+list_data)
        # else:
        #     st.write("Uploaded Data:")

        st.write(df)
        time.sleep(5)

        # Add text boxes for question and output
        
         # if st.key_press("Escape"):
            #     st.write("Escape key pressed. Exiting.")
            #     break
        #Question and Response Containers

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "Hi!!"}]
            st.session_state.messages.append({"role": "assistant", "content": "I am your personal assistant"})
            st.session_state.messages.append({"role": "assistant", "content": "How may I help you today?"})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
            


            
            
        question = st.chat_input("Ask me anything from the file uploaded")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                    st.write(question)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    result=ask_csv(df, question)
                    
                    
                # Here, you can process the question and provide an output based on the data in the CSV
                # For demonstration purposes, we'll simply display the question.
                        #st.write("Question:", question)
                    if result['type']=='dataframe':
                        result_df=result['value']
                        if isinstance(result_df,pd.Series):
                            result_df=result_df.to_frame().reset_index()
                            
                                                    
                        numeric_cols=result_df.select_dtypes(include=[np.int64,np.int, np.float64,np.number]).columns.tolist()
                    
                        column_config=generate_column_config(result_df) 
                        if len(numeric_cols)>0:
                            result_df.sort_values(by=numeric_cols,inplace=True,ascending=False)

                            #print(column_config)   
                        try:                       
                            st.dataframe(result_df,column_config=column_config,hide_index=True)
                        except:
                            st.dataframe(result_df,hide_index=True)
                        generate_charts(result_df)
                        response_message = {"role": "assistant", "content": result_df}
                        st.session_state.messages.append(response_message)    



                    elif result['type']=='number'  or result['type']== 'text':
                        st.markdown(result['value'])
                        message = {"role": "assistant", "content": result['value']}
                        st.session_state.messages.append(message)    

                                    
                    elif result['type']=='plot':
                            st.image(result['value'])
                            message = {"role": "assistant", "content": result['value']}
                            st.session_state.messages.append(message)    

                                    
                    else:
                        st.write("Invalid results")
                        message = {"role": "assistant", "content": "Invalid results"}
                        st.session_state.messages.append(message)    

                            
            
    # Display information about what the tool can and cannot do on the right side
        st.sidebar.title("Tool Information")
        st.sidebar.markdown(''':red[### What This Tool Can Do:]''')
        st.sidebar.write("- Upload CSV or Excel files.")
        st.sidebar.write("- Enter questions and get answers based on the file uploaded.")

        st.sidebar.markdown(''':red[### What This Tool Cannot Do:]''')
        st.sidebar.write("- Handle complex data processing.")
        st.sidebar.write("- Advanced natural language processing.")
        st.sidebar.write("- Answer Questions out of the file")
                



# data = {
#     'member_id': [1, 2, 3, 4, 5,1, 2, 3, 4, 5],
#     'claim_id': [101, 102, 103, 104, 105,106,107,108,109,110],
#     'Provider_name': ['Provider A', 'Provider B', 'Provider C', 'Provider D', 'Provider E','Provider A', 'Provider B', 'Provider C', 'Provider D', 'Provider E'],
#     'icd10code': ['A00', 'B01', 'C02', 'D03', 'E04','A00', 'B01', 'C02', 'D03', 'E04'],
#     'claim_amount': [500.0, 750.0, 600.0, 800.0, 900.0,1000,2000,1500,1200,1100]
# }

# # Create a Pandas DataFrame
# df = pd.DataFrame(data)

# question='Show me the comparison of total claims for each ICD10code using graph'
# code_generator=generate_code(df)
# prompt=code_generator.prompt_generation2(question=question)
# code=code_generator.generate(prompt)
