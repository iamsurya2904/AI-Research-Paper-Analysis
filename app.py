import os
import asyncio
import time
import streamlit as st
import google.generativeai as genai  

import aiohttp  
import pandas as pd  
import pickle
import numpy as np
from pathlib import Path
import plotly.express as px  

from LLMSelect import LLMSelector
from asyncExplainer import AsyncExplanationGenerator
from preprocessor import PaperPreprocessor

# Set the GROQ_API_KEY environment variable
os.environ['GROQ_API_KEY'] = ''

class StreamlitApp:
    def __init__(self):
        st.set_page_config(page_title="Edusmart AI - Research Paper Analyzer", layout="wide")
        self.set_custom_css()
        # Ensure session state is initialized for feature_tab
        if "feature_tab" not in st.session_state:
            st.session_state["feature_tab"] = "Research Paper Analyzer"

    def set_custom_css(self):
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        :root {
            --background-color: #f0f4f8;
            --text-color: #1a1a1a;
            --card-background: #ffffff;
            --button-color: #4CAF50;
            --button-hover-color: #45a049;
            --border-color: #e0e0e0;
            --header-color: #2c3e50;
            --subheader-color: #34495e;
            --alert-background: #e3f2fd;
            --alert-text: #0d47a1;
            --paper-summary-background: #e8f5e9;
            --paper-summary-border: #4CAF50;
            --paper-summary-text: #1a1a1a; /* Ensure text color is set */
            --chunk-header-background: #3498db;
            --chunk-header-text: white;
            --section-separator: #d1d5db;
        }

        .dark-theme {
            --background-color: #1e1e1e;
            --text-color: #f0f0f0;
            --card-background: #2d2d2d;
            --button-color: #388e3c;
            --button-hover-color: #2e7d32;
            --border-color: #4a4a4a;
            --header-color: #bb86fc;
            --subheader-color: #03dac6;
            --alert-background: #1a237e;
            --alert-text: #8c9eff;
            --paper-summary-background: #1b5e20;
            --paper-summary-border: #4CAF50;
            --paper-summary-text: #f0f0f0; /* Ensure text color is set */
            --chunk-header-background: #1565c0;
            --chunk-header-text: #e3f2fd;
            --section-separator: #4a4a4a;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .main > div {
            padding-top: 2rem;
        }
        
        .stButton > button {
            width: 100%;
            background-color: var(--button-color);
            color: var(--chunk-header-text);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        
        .stButton > button:hover {
            background-color: var(--button-hover-color);
        }
        
        .stTextInput > div > div > input {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-color);
        }
        
        .stSelectbox > div > div > select {
            background-color: var (--card-background);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-color);
        }
        
        h1 {
            color: var(--header-color);
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        h2 {
            color: var(--subheader-color);
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .stAlert > div {
            padding: 0.75rem 1rem;
            border-radius: 4px;
            background-color: var(--alert-background);
            color: var(--alert-text);
        }
        
        .chunk-text, .explanation-text {
            background-color: var(--card-background);
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            color: var(--text-color); /* Ensure text color is set */
        }
        
        .sidebar .stRadio > div {
            background-color: var(--card-background);
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }
        
        .stProgress > div > div > div > div {
            background-color: var(--button-color);
        }
        
        .paper-summary {
            background-color: var(--paper-summary-background);
            border-left: 5px solid var(--paper-summary-border);
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: var(--paper-summary-text); /* Ensure text color is set */
        }
        
        .chunk-header {
            background-color: var(--chunk-header-background);
            color: var(--chunk-header-text);
            padding: 0.5rem 1rem;
            border-radius: 4px 4px 0 0;
            margin-bottom: 0;
        }

        .section-separator {
            border-top: 1px solid var(--section-separator);
            margin: 1rem 0;
        }
        
        </style>
        """, unsafe_allow_html=True)

    def sidebar_content(self):
        with st.sidebar:
            st.title("Configuration")
            
            uploaded_file = st.file_uploader('Upload Research Paper (PDF)', type='pdf')
            
            model_lst = [
                'llama3-70b-8192',  # Supported model
                'llama3-8b-8192',   # Supported model
                'mixtral-8x7b-32768-groq',
                'gemma-7b-it',
                'gemma2-9b-it',
                'gemini-pro',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
                'gpt-4',
                'gpt-3.5-turbo',
                'chatgpt-4o-latest',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307',
                'mistral-large-2402',
                'mistral-large-2407',
            ]
        
            model_name = st.selectbox('Choose Model', model_lst)
            name_to_api_provider = {
                'llama3-70b-8192': 'groq',  # Updated mapping
                'llama3-8b-8192': 'groq',   # Updated mapping
                'mixtral-8x7b-32768-groq': 'groq',
                'gemma-7b-it': 'groq',
                'gemma2-9b-it': 'groq',
                'gemini-pro': 'google',
                'gemini-1.5-pro': 'google',
                'gemini-1.5-flash': 'google',
                'gpt-4o': 'openai',
                'gpt-4o-mini': 'openai',
                'gpt-4-turbo': 'openai',
                'gpt-4': 'openai',
                'gpt-3.5-turbo': 'openai',
                'chatgpt-4o-latest': 'openai',
                'claude-3-opus-20240229': 'anthropic',
                'claude-3-sonnet-20240229': 'anthropic',
                'claude-3-haiku-20240307': 'anthropic',
                'mistral-large-2402': 'mistralai',
                'mistral-large-2407': 'mistralai'
            }
            
            st.subheader("Explanation Options")
            options = {
                "difficulty": st.select_slider("Difficulty Level", ["High School", "Undergraduate", "Graduate", "Expert"]),
                "include_examples": st.checkbox("Include Examples", value=False),
                "explain_prereq": st.checkbox("Explain Prerequisites", value=False),
                "explain_math": st.checkbox("Explain Mathematical Concepts", value=False),
                "find_similar_papers": st.checkbox("Find Similar Papers", value=False, disabled=True)
            }
            
            if options["find_similar_papers"]:
                options["include_paper_summary"] = st.checkbox("Include Summary of Similar Papers", value=False, disabled=True)
            else:
                options["include_paper_summary"] = False
                st.info("'Find Similar Papers' feature is currently under development and will be available soon. Stay tuned!")
            
            with st.expander("Additional Options"):
                execution_mode = st.radio("Execution Mode", ["Async", "Non-Async"])
                
                if execution_mode == "Non-Async":
                    options["sleep_between"] = st.checkbox("Include Sleeps", value=False)
                else:
                    options["sleep_between"] = False
                
                summarization_method = st.selectbox("Summarization Method", ["map_reduce", "refine", "stuff"])
            
            st.sidebar.markdown("---")
            # Use on_change callback to update session state
            st.radio(
                "Select Feature", 
                ["Research Paper Analyzer", "Career Guidance System"], 
                key="feature_tab",
                on_change=self.update_feature_tab
            )

            return uploaded_file, model_name, name_to_api_provider[model_name], options, execution_mode, summarization_method

    def update_feature_tab(self):
        # Callback to handle feature tab changes
        st.session_state["feature_tab"] = st.session_state.feature_tab

    def main_content(self, uploaded_file, model_name, api_provider, options, execution_mode, summarization_method):
        selected_tab = st.session_state.get("feature_tab", "Research Paper Analyzer")
        if selected_tab == "Research Paper Analyzer":
            self.research_paper_analyzer(uploaded_file, model_name, api_provider, options, execution_mode, summarization_method)
        elif selected_tab == "Career Guidance System":
            self.career_guidance_system()

    def research_paper_analyzer(self, uploaded_file, model_name, api_provider, options, execution_mode, summarization_method):
        st.title('Edusmart AI - Research Paper Summarizer')
        st.write("Upload a research paper PDF and get an explanation tailored to your needs.")

        if uploaded_file:
            if st.button('Process Paper'):
                if execution_mode == "Async":
                    asyncio.run(self.process_paper_async(uploaded_file, model_name, api_provider, options, summarization_method))
                else:
                    self.process_paper_sync(uploaded_file, model_name, api_provider, options, summarization_method)
        elif not uploaded_file:
            st.info('Please upload a PDF file in the sidebar to begin.')

    def load_career_resources(self):
        """Load model and dataset with proper path handling and data conversion"""
        try:
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / "careerlast.pkl"
            dataset_path = base_dir / "dataset9000.csv"

            model = pickle.load(open(model_path, 'rb'))
            career_data = pd.read_csv(dataset_path, header=None)
            
            # Convert text values to numeric scores (0-100)
            skill_mapping = {
                'Not Interested': 0,
                'Poor': 25,
                'Beginner': 40,
                'Average': 60,
                'Professional': 85,
                'Expert': 100
            }
            
            # Apply mapping to convert text to numbers
            career_data = career_data.replace(skill_mapping)
            
            # Set column names (same as before)
            career_data.columns = [
                "Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
                "Cyber Security", "Networking", "Development", "Programming Skills", 
                "Project Management", "Computer Forensics Fundamentals", "Technical Communication",
                "AI ML", "Software Engineering", "Business Analysis", "Communication skills",
                "Data Science", "Troubleshooting skills", "Graphics Designing", "Roles"
            ]
            
            return model, career_data
        except Exception as e:
            st.error(f"Error loading career resources: {str(e)}")
            return None, None

    def career_guidance_system(self):
        st.title("🚀 AI Career Guidance System")
        st.markdown("Discover your ideal tech career based on skill proficiency (0-100 scale)")

        model, career_data = self.load_career_resources()
        if model is None or career_data is None:
            return

        jobs_dict = {
            0: 'AI/ML Specialist', 1: 'API Integration Specialist',
            2: 'Application Support Engineer', 3: 'Business Analyst',
            4: 'Customer Service Executive', 5: 'Cyber Security Specialist',
            6: 'Data Scientist', 7: 'Database Administrator',
            8: 'Graphics Designer', 9: 'Hardware Engineer',
            10: 'Helpdesk Engineer', 11: 'Information Security Specialist',
            12: 'Networking Engineer', 13: 'Project Manager',
            14: 'Software Developer', 15: 'Software Tester',
            16: 'Technical Writer'
        }

        with st.form("career_form"):
            cols = st.columns(4)
            skills = [
                "Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
                "Cyber Security", "Networking", "Development", "Programming Skills", 
                "Project Management", "Computer Forensics Fundamentals", "Technical Communication",
                "AI ML", "Software Engineering", "Business Analysis", "Communication skills",
                "Data Science", "Troubleshooting skills", "Graphics Designing"
            ]

            skill_inputs = []
            for i, skill in enumerate(skills):
                with cols[i % 4]:
                    skill_inputs.append(
                        st.slider(skill, 0, 100, 50, help=f"Rate your {skill} proficiency (0-100)")
                    )

            submitted = st.form_submit_button("Analyze Skills")
            
            if submitted:
                with st.spinner("Analyzing your skills profile..."):
                    try:
                        input_data = np.array(skill_inputs).reshape(1, -1)
                        raw_prediction = model.predict(input_data)[0]
                        
                        # Handle both numeric and string predictions
                        if isinstance(raw_prediction, str):
                            # Find the numeric ID for this career name
                            prediction = next((k for k, v in jobs_dict.items() if v == raw_prediction), None)
                            if prediction is None:
                                raise ValueError(f"Unknown career prediction: {raw_prediction}")
                        else:
                            prediction = raw_prediction
                        
                        # Verify prediction is valid
                        if prediction not in jobs_dict:
                            raise ValueError(f"Invalid prediction value: {prediction}")
                        
                        probabilities = model.predict_proba(input_data)[0]
                        top5_indices = np.argsort(probabilities)[::-1][:5]

                        # Display results
                        st.markdown("---")
                        st.subheader("Career Recommendations")
                        
                        # Main recommendation
                        st.markdown(f"""
                        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                            <h3 style='color:#2e7d32; font-size:24px; font-weight:bold;'>Top Career Match</h3>
                            <h2 style='color:#1b5e20; font-size:28px; font-weight:bold;'>{jobs_dict[prediction]}</h2>
                            <p style='color:#388e3c; font-size:18px;'>Confidence: <strong>{probabilities[prediction]*100:.1f}%</strong></p>
                            <a href="http://your-domain-or-ip/courses.php?role={jobs_dict[prediction].replace(' ', '_')}" target="_blank" style='color:#ffffff; background-color:#4CAF50; padding:10px 15px; text-decoration:none; border-radius:5px; font-size:16px;'>Go to Course</a>
                        </div>
                        """, unsafe_allow_html=True)

                        # Alternative careers
                        st.subheader("Alternative Career Paths")
                        alt_cols = st.columns(4)
                        for i, idx in enumerate(top5_indices[1:5]):
                            if idx in jobs_dict:
                                try:
                                    st.markdown(f"""
                                    <div style='background-color:#e3f2fd; padding:15px; border-radius:8px; margin-bottom:10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                                        <h4 style='color:#1565c0; font-size:20px; font-weight:bold;'>{jobs_dict[idx]}</h4>
                                        <p style='color:#0d47a1; font-size:16px;'>Match: <strong>{probabilities[idx]*100:.1f}%</strong></p>
                                        <a href="http://your-domain-or-ip/courses.php?role={jobs_dict[idx].replace(' ', '_')}" target="_blank" style='color:#ffffff; background-color:#1565c0; padding:8px 12px; text-decoration:none; border-radius:5px; font-size:14px;'>Go to Course</a>
                                    </div>
                                    """, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error generating link for {jobs_dict[idx]}: {str(e)}")

                        # Skill comparison (now with properly converted data)
                        st.markdown("---")
                        st.subheader("Skill Development Insights")

                        try:
                            # 1. Convert user inputs to numeric (float)
                            user_scores = [float(score) for score in skill_inputs]  # Explicit conversion
                            
                            # 2. Prepare career data - convert all values to numeric
                            skill_mapping = {
                                'Not Interested': 0,
                                'Poor': 25,
                                'Beginner': 40,
                                'Average': 60,
                                'Professional': 85,
                                'Expert': 100
                            }
                            
                            # Convert all values in career data to numeric
                            numeric_career_data = career_data.iloc[:, :17].applymap(
                                lambda x: skill_mapping.get(x, 0) if isinstance(x, str) else float(x)
                            )
                            
                            # 3. Calculate average scores
                            avg_scores = numeric_career_data.mean()
                            
                            # 4. Create comparison DataFrame with explicit type conversion
                            comparison = pd.DataFrame({
                                'Skill': pd.Series(skills, dtype='str'),
                                'Your Skills': pd.Series(user_scores, dtype='float64'),
                                'Industry Average': pd.Series(avg_scores.values, dtype='float64')
                            })
                            
                            # 5. Verify data types
                            if not all(comparison.dtypes == ['object', 'float64', 'float64']):
                                raise ValueError("Data type conversion failed")
                            
                            # 6. Create visualization
                            fig = px.bar(
                                comparison.melt(id_vars='Skill'),
                                x='Skill',
                                y='value',
                                color='variable',
                                barmode='group',
                                labels={'value': 'Proficiency (0-100)', 'variable': ''},
                                height=500
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Skill analysis error: {str(e)}")
                            st.write("Debug Info:")
                            st.write(f"User scores types: {[type(x) for x in skill_inputs]}")
                            st.write("Career data sample:", career_data.iloc[:, :17].head(2))
                            st.write("Numeric conversion sample:", numeric_career_data.head(2) if 'numeric_career_data' in locals() else "N/A")

                        # Debug: Print unique values in career data
                        print("Unique values in career data:")
                        for col in career_data.iloc[:, :17].columns:
                            print(f"{col}: {career_data[col].unique()}")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.write("Debug information:")
                        st.write(f"Raw prediction: {raw_prediction}")
                        st.write(f"Prediction type: {type(raw_prediction)}")
                        if 'probabilities' in locals():
                            st.write(f"All probabilities: {probabilities}")

    def display_career_details(self, career_id):
        """Display career details from the corresponding PHP file or provide default info"""
        career_files = {
            0: "AI_ML_Specialist.php",
            1: "API_Integration_Specialist.php",  # Corrected filename (removed space)
            2: "Application_Support_Engineer.php",
            3: "Business_Analyst.php",
            4: "Customer_Service_Executive.php",
            5: "Cyber_Security_Specialist.php",
            6: "Data_Scientist.php",
            7: "Database_Administrator.php",
            8: "Graphic_Designer.php",
            9: "Hardware_Engineer.php",
            10: "Helpdesk_Engineer.php",
            11: "Information_security.php",
            12: "Networking_engineer.php",
            13: "Project_Manager.php",
            14: "Software_developer.php",
            15: "Software_tester.php",
            16: "Technical_writer.php"
        }
        
        # Default descriptions for each career
        default_descriptions = {
            1: """
            <h3>API Integration Specialist</h3>
            <p>An API Integration Specialist designs, implements, and maintains API connections between different software systems.</p>
            
            <h4>Key Responsibilities:</h4>
            <ul>
                <li>Develop and maintain API integrations</li>
                <li>Ensure secure data exchange between systems</li>
                <li>Troubleshoot integration issues</li>
                <li>Document API specifications</li>
            </ul>
            
            <h4>Required Skills:</h4>
            <ul>
                <li>RESTful API design</li>
                <li>Authentication protocols (OAuth, JWT)</li>
                <li>Programming (Python, JavaScript, Java)</li>
                <li>API testing tools (Postman, Swagger)</li>
            </ul>
            """
        }

        try:
            file_name = career_files.get(career_id)
            if not file_name:
                raise FileNotFoundError(f"No file mapped for career ID {career_id}")
                
            file_path = Path(__file__).parent / file_name
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple extraction of content between <body> tags
                    body_content = content.split('<body>')[1].split('</body>')[0] if '<body>' in content else content
                    # Basic HTML cleaning for Streamlit
                    cleaned_content = body_content.replace('<?php', '').replace('?>', '')
                    st.markdown(cleaned_content, unsafe_allow_html=True)
            else:
                # Use default description if file doesn't exist
                if career_id in default_descriptions:
                    st.markdown(default_descriptions[career_id], unsafe_allow_html=True)
                else:
                    st.info("Detailed career information not available")
                    
        except Exception as e:
            st.warning(f"Couldn't load career details: {str(e)}")
            if career_id in default_descriptions:
                st.markdown(default_descriptions[career_id], unsafe_allow_html=True)

    async def process_paper_async(self, uploaded_file, model_name, api_provider, options, summarization_method):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Extracting text from PDF...")
        paper_text = PaperPreprocessor.extract_text_from_pdf(uploaded_file)

        status_text.text(f"Initializing {model_name} model from {api_provider}...")
        llm = LLMSelector.get_llm(api_provider, model_name)
        generator = AsyncExplanationGenerator(llm)

        progress_bar.progress(10)
        status_text.text("Summarizing the paper...")
        summary = generator.summarize_paper(paper_text, summarization_method)
        paper_chunks = generator.text_splitter.split_text(paper_text)

        progress_bar.progress(30)
        status_text.text("Generating explanations...")

        tasks = [
            self.retry_task(generator.generate_main_explanation_async(summary, paper_chunks, difficulty=options['difficulty'])),
            self.retry_task(generator.generate_examples_async(summary, paper_chunks)) if options['include_examples'] else None,
            self.retry_task(generator.explain_prerequisites_async(summary, paper_chunks)) if options["explain_prereq"] else None,
            self.retry_task(generator.explain_math_async(summary, paper_chunks)) if options["explain_math"] else None,
            self.retry_task(generator.find_similar_papers_async(summary, options["include_paper_summary"])) if options["find_similar_papers"] else None
        ]
        results = await asyncio.gather(*[task for task in tasks if task is not None])

        explanations = {
            'Prerequisites': results[2] if options["explain_prereq"] else None,
            'Main Explanation': results[0],
            'Examples': results[1] if options['include_examples'] else None,
            'Mathematical Concepts': results[3] if options["explain_math"] else None,
            'Similar Papers': results[4] if options["find_similar_papers"] else None
        }
        explanations = {k: v for k, v in explanations.items() if v is not None}

        progress_bar.progress(90)
        status_text.text("Combining explanations...")
        final_explanation = {
            'summary': summary,
            'explanations': generator.combine_explanations(summary, explanations, paper_chunks)
        }

        progress_bar.progress(100)
        status_text.text("Displaying results...")
        self.display_results(summary, final_explanation)
        status_text.text('Process Completed!')

    async def retry_task(self, task, retries=5, backoff_factor=2):
        for attempt in range(retries):
            try:
                return await task
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit error
                    wait_time = backoff_factor ** attempt
                    st.warning(f"Rate limit reached. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        raise Exception("Max retries exceeded")

    def process_paper_sync(self, uploaded_file, model_name, api_provider, options, summarization_method):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Extracting text from PDF...")
        paper_text = PaperPreprocessor.extract_text_from_pdf(uploaded_file)

        status_text.text(f"Initializing {model_name} model from {api_provider}...")
        llm = LLMSelector.get_llm(api_provider, model_name)
        generator = AsyncExplanationGenerator(llm)

        progress_bar.progress(10)
        status_text.text("Summarizing the paper...")
        summary = generator.summarize_paper(paper_text, summarization_method)

        if options["sleep_between"]:
            status_text.text("Sleeping for 60 seconds...")
            time.sleep(60)

        paper_chunks = generator.text_splitter.split_text(paper_text)

        progress_bar.progress(30)
        status_text.text("Generating explanations...")
        explanations = {}
        
        explanations['Main Explanation'] = generator.generate_main_explanation_sync(summary, paper_chunks, difficulty=options['difficulty'])

        if options["sleep_between"]:
            status_text.text("Sleeping for 60 seconds...")
            time.sleep(60)

        progress_bar.progress(50)
        
        if options['include_examples']:
            explanations['Examples'] = generator.generate_examples_sync(summary, paper_chunks)
            if options["sleep_between"]:
                status_text.text("Sleeping for 60 seconds...")
                time.sleep(60)

        progress_bar.progress(60)
        
        if options["explain_prereq"]:
            explanations['Prerequisites'] = generator.explain_prerequisites_sync(summary, paper_chunks)
            if options["sleep_between"]:
                status_text.text("Sleeping for 60 seconds...")
                time.sleep(60)
        progress_bar.progress(70)
        
        if options["explain_math"]:
            explanations['Mathematical Concepts'] = generator.explain_math_sync(summary, paper_chunks)
            if options["sleep_between"]:
                status_text.text("Sleeping for 60 seconds...")
                time.sleep(60)
        progress_bar.progress(80)
        
        if options["find_similar_papers"]:
            explanations['Similar Papers'] = generator.find_similar_papers_sync(summary, options["include_paper_summary"])
            if options["sleep_between"]:
                status_text.text("Sleeping for 60 seconds...")
                time.sleep(60)

        progress_bar.progress(90)
        status_text.text("Combining explanations...")
        final_explanation = {
            'summary': summary,
            'explanations': generator.combine_explanations(summary, explanations, paper_chunks)
        }

        progress_bar.progress(100)
        status_text.text("Displaying results...")
        self.display_results(summary, final_explanation)
        status_text.text('Process Completed!')

    def analyze_with_gemini(self, content):
        genai.configure(api_key="AIzaSyC_3IKqgOMJ1FUyCW8g2JRXa6tI1EJn0Iw")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(content)
        return response.text

    def display_results(self, summary, final_explanation):
        st.success('Paper processed successfully!')
        
        st.markdown("## Paper Summary")
        st.markdown(f'<div class="paper-summary">{final_explanation["summary"]}</div>', unsafe_allow_html=True)
        
        chunks = final_explanation['explanations'].split("## Chunk")
        for i, chunk in enumerate(chunks[1:], 1):
            st.markdown(f'<h2 class="chunk-header">Chunk {i}</h2>', unsafe_allow_html=True)
            
            sections = chunk.split("###")
            chunk_content = sections[0].strip()
            if chunk_content:
                st.markdown(f'<div class="chunk-text">{chunk_content}</div>', unsafe_allow_html=True)
            
            for section in sections[1:]:
                section_parts = section.split("\n", 1)
                if len(section_parts) == 2:
                    section_title, section_content = section_parts
                    st.markdown(f'<h3>{section_title.strip()}</h3>', unsafe_allow_html=True)
                    st.markdown(f'<div class="explanation-text">{section_content.strip()}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)

        explanations = final_explanation['explanations']
        
        if 'Prerequisites' in explanations:
            st.markdown("## Prerequisites")
            st.markdown(f'<div class="explanation-text">{explanations["Prerequisites"]}</div>', unsafe_allow_html=True)

        if 'Examples' in explanations:
            st.markdown("## Examples")
            st.markdown(f'<div class="explanation-text">{explanations["Examples"]}</div>', unsafe_allow_html=True)

        if 'Mathematical Concepts' in explanations:
            st.markdown("## Mathematical Concepts")
            st.markdown(f'<div class="explanation-text">{explanations["Mathematical Concepts"]}</div>', unsafe_allow_html=True)

        # Add section for further advancements and suggestions
        st.markdown("## Further Advancements and Suggestions")
        st.markdown("""
        Based on the analysis and explanations provided, here are some suggestions for further advancements and improvements:
        - **Expand Research Scope**: Consider exploring additional related topics to broaden the scope of the research.
        - **Incorporate Latest Technologies**: Utilize the latest advancements in technology to enhance the research outcomes.
        - **Collaborate with Experts**: Engage with experts in the field to gain deeper insights and refine the research methodology.
        - **Improve Data Collection**: Enhance the data collection process to ensure more accurate and comprehensive results.
        - **Conduct Comparative Studies**: Perform comparative studies with similar research to identify strengths and areas for improvement.
        - **Seek Peer Reviews**: Obtain feedback from peers to identify potential gaps and areas for enhancement.
        - **Publish Findings**: Share the research findings in reputable journals and conferences to gain wider recognition and feedback.
        """, unsafe_allow_html=True)

        # Analyze with Gemini AI
        st.markdown("## Analysis by Gemini AI")
        gemini_analysis_content = f"""
        Provide suggestions based on the following:
        Paper Summary: {final_explanation["summary"]}
        Chunk 1: {chunks[1] if len(chunks) > 1 else "N/A"}
        Explanation Sections: {explanations}
        """
        gemini_analysis = self.analyze_with_gemini(gemini_analysis_content)
        st.markdown(f'<div class="explanation-text">{gemini_analysis}</div>', unsafe_allow_html=True)

    def run(self):
        uploaded_file, model_name, api_provider, options, execution_mode, summarization_method = self.sidebar_content()
        self.main_content(uploaded_file, model_name, api_provider, options, execution_mode, summarization_method)

if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
