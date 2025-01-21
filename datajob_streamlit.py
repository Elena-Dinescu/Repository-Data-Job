import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("kaggle_survey_2020_responses.csv")

st.title("Data Industry Role Recommender System")
st.sidebar.title("Table of contents")
st.sidebar.write("Introduction")
pages=["Introduction", "Data Understanding", "Data Exploration","Data Transformation", "Model Training and Evaluation"]
page=st.sidebar.radio("Go to", pages)


if page == pages[0] : 
  st.write("### Context")
  st.write("""
  The rapid expansion of the data industry has given rise to a variety of technical roles, each requiring specific skill sets and expertise. 
  This diversification reflects the increasing demand for data-driven solutions across industries, necessitating a better understanding of these roles and their interdependencies.

  ### Key Positions:
  - **Data Analyst**: Focuses on interpreting data to provide actionable insights, often utilizing tools for data visualization and statistical analysis.
  - **Data Scientist**: Engages in building predictive models and performing complex analyses, requiring proficiency in programming languages and machine learning techniques.
  - **Data Engineer**: Responsible for designing and maintaining data pipelines and architectures, ensuring data is accessible and reliable for analysis.
  - **Machine Learning Engineer**: Specializes in developing and optimizing machine learning algorithms to make accurate predictions based on historical data.
  - **Data Visualization Specialist**: Creates visually engaging representations of data to facilitate understanding and decision-making.

  ### About the Dataset
  The data analyzed in this project originates from the 2020 Kaggle Machine Learning & Data Science Survey. Kaggle, under Google LLC, is a prominent platform and online community for data science and machine learning practitioners. 

  The survey captures insights into tools, tasks, and compensation across a wide spectrum of roles in the data industry. This dataset is particularly valuable as it represents a global perspective, with responses from professionals across various countries, industries, and experience levels.

  ### Objectives
  The primary goal of this project is to analyze tasks and tools used in different technical roles within the data industry with the purpose of:
  - Establish the range of skills required for each role.
  - Create a role recommender system to guide individuals in aligning their skills and aspirations with suitable roles in data science
           
  Recommender systems are a prominent application within data science, utilizing machine learning techniques to analyze user data and generate personalized suggestions. These systems are integral to various industries, enhancing user experience by tailoring content and product recommendations to individual preferences. By examining the survey's insights into programming languages, tools, and methodologies favored by data scientists, one can identify the competencies essential for building effective recommender systems. We will test our assumptions with statistics and showcase using visuals our findings and interesting patterns. To start, we’ll focus on gaining a deeper understanding of the dataset and the survey’s context.
           
  ### Framework
  The survey was conducted over 3.5 weeks in October 2020, resulting in 20,036 clean responses grouped in 355 columns. This dataset reveals insights about who works with data, how machine learning is being used across industries, and the best strategies for aspiring data scientists to enter the field. It was made available in its rawest possible form while ensuring anonymity and it is freely accessible on the Kaggle website.
                      
  """)
  


if page == pages[1] : 
  st.write("""### Understanding the Data
  The 2020 Kaggle DS & ML Survey received 20,036 usable responses from participants in 171 different countries and territories. If a country or territory received less than 50 respondents, they were put into a group named “Other” for anonymity.

  The 2020 Kaggle DS & ML Survey received 20,036 usable responses from participants in 171 different countries and territories. If a country or territory received less than 50 respondents, they were put into a group named “Other” for anonymity.

  To protect the respondents’ privacy, free-form text responses were not included in the public survey dataset, and the order of the rows was shuffled (responses are not displayed in chronological order).

  The data includes demographic information (e.g., age, gender, location) and detailed responses about roles, tools, and programming languages.     

  **Survey Flow Logic**:

  - Respondents with the most experience were asked the most questions. For example, students and unemployed people were not asked questions about their employer. Likewise, respondents that do not write code were not asked questions about writing code.
  - For questions about cloud computing products, students and respondents that have never spent money in the cloud were given an alternate set of questions that asked them “what products they would like to become familiar with” instead of asking them “which products they use most often”.
  - For questions with alternative phrasing, the questions were kept separate, and question types were labeled with either an “A” or a “B” (e.g. Q25A, Q25B, … , Q34A, Q34B).
  - Follow-up questions were only asked to respondents that answered the setup question affirmatively:
      - Question 18 and Question 19 (which specific ML methods) were only asked to respondents that selected the relevant answer choices for Question 17 (which categories of algorithms).
      - Question 27-A and Question 28-A (which specific AWS/Azure/GCP products) were only asked to respondents that selected the relevant answer choices for Question 26-A (which of the following companies).
      - Question 30 (which specific product) was only asked to respondents that selected more than one choice for Question 29-A (which of the following products).
      - Question 32 (which specific product) was only asked to respondents that selected more than one choice for Question 31-A (which of the following products).
      - Question 34-A (which specific product) was only asked by respondents that answered affirmatively Question 33-A (which of the following categories of products).
           
   ### Data Structure

  The dataset contains 355 columns, most of them related to survey responses. Except for the **duration of the survey**, the dataset features only categorical data. Columns correspond to survey questions and rows represent individual responses.
           """) 
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())

  if st.checkbox("Show NA") :
    st.dataframe(df.isna().sum())

  st.write("""
  The first row contains the question’s text for each column (metadata), and the actual responses start from the second row onward. The first column contains the duration of the survey per participant. We can safely remove both the first column from the dataset. Going further with the analysis, 19 rows are duplicates. We will delete those also.
  
  The **Column Q5**, which contains responses to the question, "Select the title most similar to your current role (or most recent title if retired)," will serve as the target variable in this machine learning scenario. The remaining variables will be used as features. (For a detailed breakdown of the variables, refer to the Data Audit Template.)

  Rows with missing values in the Q5 column were removed, as these rows exclusively contain NaN values across all columns except for Q1 to Q4.
           
  ### Data Limitations
  The dataset includes several columns with significant amounts of missing data, such as those capturing tool usage and compensation. For instance, columns like Q24 (yearly compensation) have a substantial proportion of missing values, which limits the ability to draw reliable insights into financial trends. Furthermore, the extent of missingness varies across demographic groups, potentially introducing bias in analysis. Addressing these gaps through imputation or filtering is critical to ensuring robust results.

  Participation was voluntary, which could result in a sample that doesn't accurately reflect the broader community. The dataset may overrepresent respondents from countries with higher internet penetration or regions where Kaggle's platform is more widely used. Since the survey was conducted on Kaggle, the results may be more reflective of Kaggle's user base, which might not represent data scientists who don't engage with the platform.

  Only 1 in 5 survey participants is a woman. This imbalance can lead to skewed insights regarding technical roles in the data industry, potentially misrepresenting the tasks, tools, and skills associated with each position. The analysis may overemphasize roles and skills more prevalent or preferred among men, while underrepresenting those associated with women. This can result in an incomplete understanding of the data industry's landscape. A role recommender system trained on gender-biased data might suggest career paths aligning with the majority gender's responses, thereby perpetuating existing gender disparities in the field. The result could unintentionally reinforce stereotypes by associating certain technical roles or skills predominantly with one gender, influencing perceptions and aspirations of future professionals.

  The survey indicates that a significant portion of respondents are relatively new to the field, with over half having less than three years of machine learning experience. This could influence findings related to industry practices and tool adoption. These biases could impact the generalizability of the findings to the global data science community.

  """) 
  
if page == pages[2] : 
  st.write("""### Exploring the Data
   Visualize the distribution of respondents across different roles to understand the sample composition:
           

  Students comprised a notable segment of respondents, reflecting the growing interest and involvement in data science among individuals at the early stages of their careers. Including students in our analysis can offer valuable insights into emerging trends and educational focuses within the data industry. However, there are several considerations to keep in mind:
  - Limited Professional Experience: Students may lack practical experience, leading to responses that are more theoretical than reflective of real-world industry practices.
  - Incomplete Skill Sets: Their skill sets might be in development, potentially skewing analyses that aim to map comprehensive competencies required for various roles.
  - Aspirational vs. Actual Roles: Students' perceptions of roles may be based on aspirations or academic exposure, which might not align with industry realities.

  By carefully considering these factors, we made the decisions **not** to incorporate student data into our analysis, ensuring the relevance and accuracy of our project's outcomes.
           
  """)   
  
  
if page == pages[3] : 
  st.write("""### Transforming the Data
   The columns generated by the answers to the Multiple-Selection Questions are quite a challenge.These questions are represented across multiple columns, each indicating a possible choice. Such a structure makes it difficult to analyze the frequency of each option, compare trends, or generate insights. To analyze them effectively we will transform the wide-format data into a long-format where each row represents a single response-option pair.  It allows for easier aggregation, visualization, statistical analysis, and machine learning preprocessing. By converting data into a standardized, long format, analysts gain flexibility and insight, ultimately leading to more informed decision-making.
   """)   