import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import warnings
warnings.filterwarnings('ignore')
import signal
from contextlib import contextmanager
from langchain.llms import OpenAI
import platform

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    if platform.system() == 'Windows':
        # On Windows, just yield without timeout
        yield
    else:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)




def streamlit_config():

    
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")


    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)
    st.markdown(f'<h1 style="text-align: center;">Resume Analyzer AI</h1>',
                unsafe_allow_html=True)


class resume_analyzer:

    @staticmethod
    def pdf_to_chunks(pdf):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)

        chunks = text_splitter.split_text(text=text)
        return chunks


    @staticmethod
    def openai(openai_api_key, chunks, analyze):
        try:
            # Add retry mechanism
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    vectorstores = FAISS.from_texts(chunks, embedding=embeddings)
                    docs = vectorstores.similarity_search(query=analyze, k=3)
                    
                    llm = ChatOpenAI(
                        temperature=0,
                        model_name='gpt-3.5-turbo',
                        openai_api_key=openai_api_key
                    )
                    
                    chain = load_qa_chain(llm=llm, chain_type='stuff')
                    response = chain.run(input_documents=docs, question=analyze)
                    return response
                
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        retry_count += 1
                        time.sleep(10)  # Wait 20 seconds before retrying
                    else:
                        raise e
                        
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return None


    @staticmethod
    def summary_prompt(query_with_chunks):

        query = f''' need to detailed summarization of below resume and finally conclude them

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    @staticmethod
    def resume_summary():

        with st.form(key='Summary'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Enter OpenAI API Key
            col1,col2 = st.columns([0.6,0.4])
            with col1:
                openai_api_key = st.text_input(label='Enter OpenAI API Key', type='password')
            add_vertical_space(2)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        add_vertical_space(3)
        if submit:
            if pdf is not None and openai_api_key != '':
                try:
                    with st.spinner('Processing...'):

                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=summary_prompt)

                    st.markdown(f'<h4 style="color: orange;">Summary:</h4>', unsafe_allow_html=True)
                    st.write(summary)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif openai_api_key == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Enter OpenAI API Key</h5>', unsafe_allow_html=True)


    @staticmethod
    def strength_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    @staticmethod
    def resume_strength():

        with st.form(key='Strength'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Enter OpenAI API Key
            col1,col2 = st.columns([0.6,0.4])
            with col1:
                openai_api_key = st.text_input(label='Enter OpenAI API Key', type='password')
            add_vertical_space(2)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)

        add_vertical_space(3)
        if submit:
            if pdf is not None and openai_api_key != '':
                try:
                    with st.spinner('Processing...'):
                    
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=summary_prompt)
                        
                        strength_prompt = resume_analyzer.strength_prompt(query_with_chunks=summary)

                        strength = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=strength_prompt)

                    st.markdown(f'<h4 style="color: orange;">Strength:</h4>', unsafe_allow_html=True)
                    st.write(strength)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif openai_api_key == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Enter OpenAI API Key</h5>', unsafe_allow_html=True)


    @staticmethod
    def weakness_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the weakness of below resume and how to improve make a better resume.

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    @staticmethod
    def resume_weakness():

        with st.form(key='Weakness'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Enter OpenAI API Key
            col1,col2 = st.columns([0.6,0.4])
            with col1:
                openai_api_key = st.text_input(label='Enter OpenAI API Key', type='password')
            add_vertical_space(2)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        add_vertical_space(3)
        if submit:
            if pdf is not None and openai_api_key != '':
                try:
                    with st.spinner('Processing...'):
                    
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=summary_prompt)

                        weakness_prompt = resume_analyzer.weakness_prompt(query_with_chunks=summary)

                        weakness = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=weakness_prompt)

                    st.markdown(f'<h4 style="color: orange;">Weakness and Suggestions:</h4>', unsafe_allow_html=True)
                    st.write(weakness)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif openai_api_key == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Enter OpenAI API Key</h5>', unsafe_allow_html=True)


    @staticmethod
    def job_title_prompt(query_with_chunks):

        query = f''' what are the job roles i apply to likedin based on below?
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    @staticmethod
    def job_title_suggestion():

        with st.form(key='Job Titles'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Enter OpenAI API Key
            col1,col2 = st.columns([0.6,0.4])
            with col1:
                openai_api_key = st.text_input(label='Enter OpenAI API Key', type='password')
            add_vertical_space(2)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)

        add_vertical_space(3)
        if submit:
            if pdf is not None and openai_api_key != '':
                try:
                    with st.spinner('Processing...'):
                    
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=summary_prompt)

                        job_title_prompt = resume_analyzer.job_title_prompt(query_with_chunks=summary)

                        job_title = resume_analyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=job_title_prompt)

                    st.markdown(f'<h4 style="color: orange;">Job Titles:</h4>', unsafe_allow_html=True)
                    st.write(job_title)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif openai_api_key == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Enter OpenAI API Key</h5>', unsafe_allow_html=True)



class linkedin_scraper:

    @staticmethod
    def webdriver_setup():
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # Add these options to improve performance
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-notifications')
        options.page_load_strategy = 'eager'  # Don't wait for all resources to load
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(10)  # Set timeout to avoid hanging
        driver.maximize_window()
        return driver


    @staticmethod
    def get_userinput():

        add_vertical_space(2)
        with st.form(key='linkedin_scarp'):

            add_vertical_space(1)
            col1,col2,col3 = st.columns([0.5,0.3,0.2], gap='medium')
            with col1:
                job_title_input = st.text_input(label='Job Title')
                job_title_input = job_title_input.split(',')
            with col2:
                job_location = st.text_input(label='Job Location', value='India')
            with col3:
                job_count = st.number_input(label='Job Count', min_value=1, value=1, step=1)

            # Submit Button
            add_vertical_space(1)
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        return job_title_input, job_location, job_count, submit


    @staticmethod
    def build_url(job_title, job_location):

        b = []
        for i in job_title:
            x = i.split()
            y = '%20'.join(x)
            b.append(y)

        job_title = '%2C%20'.join(b)
        link = f"https://in.linkedin.com/jobs/search?keywords={job_title}&location={job_location}&locationId=&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"

        return link
    

    @staticmethod
    def open_link(driver, link):

        while True:
            try:
                driver.get(link)
                driver.implicitly_wait(5)
                time.sleep(3)
                driver.find_element(by=By.CSS_SELECTOR, value='span.switcher-tabs__placeholder-text.m-auto')
                return
            except NoSuchElementException:
                continue


    @staticmethod
    def link_open_scrolldown(driver, link, job_count):
        linkedin_scraper.open_link(driver, link)
        for i in range(0,job_count):
            body = driver.find_element(by=By.TAG_NAME, value='body')
            body.send_keys(Keys.PAGE_UP)
            try:
                driver.find_element(by=By.CSS_SELECTOR, 
                                value="button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']>icon>svg").click()
            except:
                pass
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.implicitly_wait(2)
            try:
                x = driver.find_element(by=By.CSS_SELECTOR, value="button[aria-label='See more jobs']").click()
                driver.implicitly_wait(5)
            except:
                pass


    @staticmethod
    def job_title_filter(scrap_job_title, user_job_title_input):
        user_input = [i.lower().strip() for i in user_job_title_input]
        scrap_title = [i.lower().strip() for i in [scrap_job_title]]
        confirmation_count = 0
        for i in user_input:
            if all(j in scrap_title[0] for j in i.split()):
                confirmation_count += 1
        if confirmation_count > 0:
            return scrap_job_title
        else:
            return np.nan


    @staticmethod
    def scrap_company_data(driver, job_title_input, job_location):
        company = driver.find_elements(by=By.CSS_SELECTOR, value='h4[class="base-search-card__subtitle"]')
        company_name = [i.text for i in company]

        location = driver.find_elements(by=By.CSS_SELECTOR, value='span[class="job-search-card__location"]')
        company_location = [i.text for i in location]

        title = driver.find_elements(by=By.CSS_SELECTOR, value='h3[class="base-search-card__title"]')
        job_title = [i.text for i in title]

        url = driver.find_elements(by=By.XPATH, value='//a[contains(@href, "/jobs/")]')
        website_url = [i.get_attribute('href') for i in url]
        df = pd.DataFrame(company_name, columns=['Company Name'])
        df['Job Title'] = pd.DataFrame(job_title)
        df['Location'] = pd.DataFrame(company_location)
        df['Website URL'] = pd.DataFrame(website_url)
        df['Job Title'] = df['Job Title'].apply(lambda x: linkedin_scraper.job_title_filter(x, job_title_input))
        df['Location'] = df['Location'].apply(lambda x: x if job_location.lower() in x.lower() else np.nan)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)

        return df 
        

    @staticmethod
    def scrap_job_description(driver, df, job_count):
        website_url = df['Website URL'].tolist()
        job_description = []
        description_count = 0

        for i in range(0, len(website_url)):
            try:
                # Add timeout to avoid hanging
                with timeout(seconds=15):
                    linkedin_scraper.open_link(driver, website_url[i])
                    driver.find_element(by=By.CSS_SELECTOR, 
                                     value='button[data-tracking-control-name="public_jobs_show-more-html-btn"]').click()
                    time.sleep(0.5)  # Reduced sleep time
                    description = driver.find_elements(by=By.CSS_SELECTOR, 
                                                    value='div[class="show-more-less-html__markup relative overflow-hidden"]')
                    data = [i.text for i in description][0]
                    
                    if len(data.strip()) > 0 and data not in job_description:
                        job_description.append(data)
                        description_count += 1
                    else:
                        job_description.append('Description Not Available')
            except:
                job_description.append('Description Not Available')
            if description_count == job_count:
                break

        df = df.iloc[:len(job_description), :]
        df['Job Description'] = pd.DataFrame(job_description, columns=['Description'])
        df['Job Description'] = df['Job Description'].apply(lambda x: np.nan if x=='Description Not Available' else x)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df


    @staticmethod
    def display_data_userinterface(df_final):
        add_vertical_space(1)
        if len(df_final) > 0:
            for i in range(0, len(df_final)):
                
                st.markdown(f'<h3 style="color: orange;">Job Posting Details : {i+1}</h3>', unsafe_allow_html=True)
                st.write(f"Company Name : {df_final.iloc[i,0]}")
                st.write(f"Job Title    : {df_final.iloc[i,1]}")
                st.write(f"Location     : {df_final.iloc[i,2]}")
                st.write(f"Website URL  : {df_final.iloc[i,3]}")

                with st.expander(label='Job Desription'):
                    st.write(df_final.iloc[i, 4])
                add_vertical_space(3)
        
        else:
            st.markdown(f'<h5 style="text-align: center;color: orange;">No Matching Jobs Found</h5>', 
                                unsafe_allow_html=True)


    @staticmethod
    def main():
        driver = None
        
        try:
            job_title_input, job_location, job_count, submit = linkedin_scraper.get_userinput()
            add_vertical_space(2)
            
            if submit:
                if job_title_input != [] and job_location != '':
                    
                    with st.spinner('Chrome Webdriver Setup Initializing...'):
                        driver = linkedin_scraper.webdriver_setup()
                                       
                    with st.spinner('Loading More Job Listings...'):

                        # build URL based on User Job Title Input
                        link = linkedin_scraper.build_url(job_title_input, job_location)

                        # Open the Link in LinkedIn and Scroll Down the Page
                        linkedin_scraper.link_open_scrolldown(driver, link, job_count)

                    with st.spinner('scraping Job Details...'):

                        # Scraping the Company Name, Location, Job Title and URL Data
                        df = linkedin_scraper.scrap_company_data(driver, job_title_input, job_location)

                        # Scraping the Job Descriptin Data
                        df_final = linkedin_scraper. scrap_job_description(driver, df, job_count)
                    
                    # Display the Data in User Interface
                    linkedin_scraper.display_data_userinterface(df_final)

                
                # If User Click Submit Button and Job Title is Empty
                elif job_title_input == []:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Title is Empty</h5>', 
                                unsafe_allow_html=True)
                
                elif job_location == '':
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', 
                                unsafe_allow_html=True)

        except Exception as e:
            add_vertical_space(2)
            st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)
        
        finally:
            if driver:
                driver.quit()



# Streamlit Configuration Setup
streamlit_config()
add_vertical_space(2)



with st.sidebar:

    add_vertical_space(4)

    option = option_menu(menu_title='', options=['Summary', 'Strength', 'Weakness', 'Job Titles', 'Linkedin Jobs'],
                         icons=['house-fill', 'database-fill', 'pass-fill', 'list-ul', 'linkedin'])



if option == 'Summary':

    resume_analyzer.resume_summary()



elif option == 'Strength':

    resume_analyzer.resume_strength()



elif option == 'Weakness':

    resume_analyzer.resume_weakness()



elif option == 'Job Titles':

    resume_analyzer.job_title_suggestion()



elif option == 'Linkedin Jobs':
    
    linkedin_scraper.main()


