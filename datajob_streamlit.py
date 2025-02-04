import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 


df=pd.read_csv("kaggle_survey_2020_responses.csv")

st.title("Data Industry Role Recommender System")
st.sidebar.title("Table of contents")
pages=["Introduction", "Data Understanding", "Data Exploration", "Model Training and Evaluation" ]
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
    st.dataframe(round((df.isnull().sum() / len(df)) * 100, 2))

  st.write("""
  The first row contains the question’s text for each column (metadata), and the actual responses start from the second row onward. The first column contains the duration of the survey per participant. We can safely remove both the first column from the dataset. Going further with the analysis, 19 rows are duplicates. We will delete those also.
  
  The **Column Q5**, which contains responses to the question, "Select the title most similar to your current role (or most recent title if retired)," will serve as the target variable in this machine learning scenario. The remaining variables will be used as features. (For a detailed breakdown of the variables, refer to the Data Audit Template.)

  Rows with missing values in the Q5 column were removed, as these rows exclusively contain NaN values across all columns except for Q1 to Q4.
           
  ### Data Limitations
  The dataset includes several columns with significant amounts of missing data, such as those capturing tool usage and compensation. For instance, columns like Q24 (yearly compensation) have a substantial proportion of missing values, which limits the ability to draw reliable insights into financial trends. Furthermore, the extent of missingness varies across demographic groups, potentially introducing bias in analysis. Addressing these gaps through imputation or filtering is critical to ensuring robust results.

  Participation was voluntary, which could result in a sample that doesn't accurately reflect the broader community. The dataset may overrepresent respondents from countries with higher internet penetration or regions where Kaggle's platform is more widely used. Since the survey was conducted on Kaggle, the results may be more reflective of Kaggle's user base, which might not represent data scientists who don't engage with the platform.

  Only 1 in 5 survey participants is a woman. This imbalance can lead to skewed insights regarding technical roles in the data industry, potentially misrepresenting the tasks, tools, and skills associated with each position. 

  The survey indicates that a significant portion of respondents are relatively new to the field, with over half having less than three years of machine learning experience. This could influence findings related to industry practices and tool adoption. These biases could impact the generalizability of the findings to the global data science community.

  """) 
  
if page == pages[2] : 
  st.write("""### Exploring the Data

   ***Multiple Choice Questions*** (only a single choice can be selected)
  """)

  columns = ['Q1', 'Q2', 'Q4', 'Q5', 'Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q20', 'Q21', 'Q24', 'Q25', 'Q30', 'Q32', 'Q38'] 
  selected_column = st.selectbox("Select a Multiple Choice Questions to visualize:", columns)

  if selected_column:
      # Skip the first row (which contains the question)
      data_for_plot = df[selected_column][1:]

      # Calculate the counts and percentages
      value_counts = data_for_plot.value_counts()
      percentages = value_counts / value_counts.sum() * 100

      # Create the countplot
      fig, ax = plt.subplots(1, 1, figsize=(15, 6))
      sns.countplot(y=data_for_plot, data=df, order=value_counts.index)

      # Add percentage annotations
      for i, p in enumerate(ax.patches):
          width = p.get_width()
          percentage = percentages[value_counts.index[i]]
          ax.annotate(f'{percentage:.1f}%', 
                      xy=(width + 50, p.get_y() + p.get_height() / 2), 
                      va='center', ha='left', fontsize=12, color='black')

      # Add a title or additional text for the question
      fig.text(0.1, 0.95, f'{df[selected_column][0]}', fontsize=12, fontweight='bold', fontfamily='serif')

      # Customize the plot
      plt.xlabel(' ', fontsize=20)
      plt.ylabel('')
      plt.yticks(fontsize=13)
      plt.box(False)

      # Show the plot
      st.pyplot(fig)

  st.write("""
   ***Multiple selection questions*** (multiple choices can be selected)
  """)

  columns = ['Q7','Q10','Q12','Q14','Q16','Q17','Q18','Q19','Q23','Q26','Q27','Q28','Q29','Q31','Q33','Q34','Q35','Q36','Q37']
  selected_column = st.selectbox("Select a Multiple selection questions to visualize:", columns) 

  if selected_column:
    # Create the figure and axis objects
    fig, ax = plt.subplots(1,1, figsize=(15, 6))

    df_q = df.iloc[1:][[i for i in df.columns if selected_column in i]]
    df_q_count = pd.Series(dtype='int')
    for i in df_q:
        value_counts = df_q[i].value_counts()
        if not value_counts.empty:
           df_q_count[df_q[i].value_counts().index[0]] = df_q[i].count()

    # Add text label to the figure
    fig.text(0.1, 0.95, f'\n\n{df[i][0].split("(")[0]}\n', fontsize=20, fontweight = 'bold', fontfamily='serif')

    # Create the bar chart using seaborn
    sns.barplot(y=df_q_count.sort_values()[::-1].index, x=df_q_count.sort_values()[::-1], ax=ax)

    # Customize the plot appearance
    plt.box(False)
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(fontsize=20)

    # Display the chart in Streamlit
    plt.box(False)
    st.pyplot(fig)



  st.write("""Visualizing the distribution of respondents across different roles we noticed that students comprised a notable segment of respondents(5171- 26.8%) , reflecting the growing interest and involvement in data science among individuals at the early stages of their careers. Including students in our analysis can offer valuable insights into emerging trends and educational focuses within the data industry. However, there are several considerations to keep in mind:
  - Limited Professional Experience: Students may lack practical experience, leading to responses that are more theoretical than reflective of real-world industry practices.
  - Incomplete Skill Sets: Their skill sets might be in development, potentially skewing analyses that aim to map comprehensive competencies required for various roles.
  - Aspirational vs. Actual Roles: Students' perceptions of roles may be based on aspirations or academic exposure, which might not align with industry realities.
         
  """) 

  if st.button('Show Programming Experience Distribution for Students'):
    # Filter the data for role = 'Student'
    student_data = df[df['Q5'] == 'Student']

    # Calculate the distribution of programming experience
    programming_exp = student_data['Q6'].value_counts().reset_index(name="Count")
    programming_exp['Percent'] = programming_exp['Count'] / programming_exp['Count'].sum() * 100
    programming_exp.columns = ['Programming Experience', 'Count', 'Percent']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Count',
        y='Programming Experience',
        data=programming_exp,
    )

    # Annotate the bars with percentages, adding space between the bar and text
    for i in range(len(programming_exp)):
        plt.annotate(
            f"{programming_exp.iloc[i]['Percent']:.1f}%",
            xy=(programming_exp.iloc[i]['Count'] + 50, i),  # Add space to the x-coordinate
            va='center',
            ha='left',
            fontsize=10
        )

    # Customize the plot
    sns.despine(left=True, bottom=True)
    plt.xlabel('Number of Participants', fontsize=12)
    plt.ylabel('Programming Experience', fontsize=12)
    plt.title('Distribution of Programming Experience for Students', fontsize=14)

    # Show the plot
    st.pyplot(plt)

  if st.button('Show Machine Learning Experience Distribution for Students'):
    # Filter the data for role = 'Student'
    student_data = df[df['Q5'] == 'Student']

    # Calculate the distribution of programming experience
    programming_exp = student_data['Q15'].value_counts().reset_index(name="Count")
    programming_exp['Percent'] = programming_exp['Count'] / programming_exp['Count'].sum() * 100
    programming_exp.columns = ['ML Experience', 'Count', 'Percent']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Count',
        y='ML Experience',
        data=programming_exp,
    )

    # Annotate the bars with percentages, adding space between the bar and text
    for i in range(len(programming_exp)):
        plt.annotate(
            f"{programming_exp.iloc[i]['Percent']:.1f}%",
            xy=(programming_exp.iloc[i]['Count'] + 50, i),  # Add space to the x-coordinate
            va='center',
            ha='left',
            fontsize=10
        )

    # Customize the plot
    sns.despine(left=True, bottom=True)
    plt.xlabel('Number of Participants', fontsize=12)
    plt.ylabel('ML Experience', fontsize=12)
    plt.title('Distribution of ML Experience for Students', fontsize=14)

    # Show the plot
    st.pyplot(plt)

  st.write(''' By carefully considering these factors, we made the decisions **not** to incorporate student data into our analysis, ensuring the relevance and accuracy of our project's outcomes.
   ''')
  #Remove the Students and work with this data frame from now on
  df_no_S= df[df['Q5'] != 'Student']

  st.write(''' ### Experience ### ''')
  st.write('''Examining programming and machine learning (ML) experience highlights the relative youth of the data science field. As shown below almost 40% of programmers have two years or less of programming experience, and more than 70% have five years or less.
 In the case of ML users, more than 70% have two years or less using ML methods. These informations, especially about ML, makes clear how new is the area.
 One reason for programming experience being bigger than ML is because some of the programmers migrated from other jobs like software development.
  ''')

  # Calculate the distribution
  if st.button('Show Programming Experience Distribution for all Roles'):
    programmingExp = df_no_S.loc[1:, ['Q6']].value_counts().reset_index(name="Count")
    programmingExp['Percent'] = programmingExp.Count / programmingExp.Count.sum() * 100
    programmingExp.columns = ['Programming Experience', 'Count', 'Percent']

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(
        x='Count',
        y='Programming Experience',
        data=programmingExp,
        # order=[x for x in order if x in programmingExp['Programming Experience'].values],
        ax=ax
    )

    # Add annotations
    for e in range(len(programmingExp)):
        ax.annotate(
            f"{programmingExp.iloc[e].Percent:.1f}%",
            xy=(bars.patches[e].get_width() + 20,  # Add space
                bars.patches[e].get_y() + (bars.patches[e].get_height() / 2)),
            va='center',
            ha='left',
            fontsize=10
        )

    # Customize plot
    ax.set_xlabel('Participants', fontsize=12)
    ax.set_ylabel('Programming Experience', fontsize=12)
    ax.set_title('Programming Experience Distribution', fontsize=14)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


  #Calculate the distribution
  if st.button('Show Machine Learning Experience Distribution for all Roles'):
    programmingExp = df_no_S.loc[1:, ['Q15']].value_counts().reset_index(name="Count")
    programmingExp['Percent'] = programmingExp.Count / programmingExp.Count.sum() * 100
    programmingExp.columns = ['Machine Learning Experience', 'Count', 'Percent']

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(
        x='Count',
        y='Machine Learning Experience',
        data=programmingExp,
        ax=ax
    )

    # Add annotations
    for e in range(len(programmingExp)):
        ax.annotate(
            f"{programmingExp.iloc[e].Percent:.1f}%",
            xy=(bars.patches[e].get_width() + 20,  # Add space
                bars.patches[e].get_y() + (bars.patches[e].get_height() / 2)),
            va='center',
            ha='left',
            fontsize=10
        )

    # Customize plot
    ax.set_xlabel('Participants', fontsize=12)
    ax.set_ylabel('Machine Learning Experience', fontsize=12)
    ax.set_title('Machine Learning Experience Distribution', fontsize=14)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

  # Define the correct Programing experience order
  experience_order = [ 'I have never written code','< 1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years']

  # Calculate the Programing experience distribution across roles
  experience_roles = df_no_S.iloc[1:].groupby(['Q5', 'Q6']).size().unstack().fillna(0)

  # Reorder columns based on Programing experience_order
  experience_roles = experience_roles.reindex(columns=experience_order, fill_value=0)

  # Create the heatmap plot
  fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis for Streamlit

  sns.heatmap(experience_roles, annot=True, fmt='.0f', cmap='Blues', cbar=True, ax=ax)

  # Customize the plot
  ax.set_title('Programming Experience Level Across Roles', fontsize=16)
  ax.set_xlabel('Programming Experience (Q15)', fontsize=14)
  ax.set_ylabel('Role (Q5)', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(fig)

    # Define the correct ML experience order
  experience_order = ['I do not use machine learning methods','Under 1 year','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-20 years','20 or more years']

  # Calculate the ML experience distribution across roles
  experience_roles = df_no_S.iloc[1:].groupby(['Q5', 'Q15']).size().unstack().fillna(0)

  # Reorder columns based on ML experience_order
  experience_roles = experience_roles.reindex(columns=experience_order, fill_value=0)

  # Create the heatmap plot
  fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis for Streamlit

  sns.heatmap(experience_roles, annot=True, fmt='.0f', cmap='Blues', cbar=True, ax=ax)

  # Customize the plot
  ax.set_title('ML Experience Level Across Roles', fontsize=16)
  ax.set_xlabel('ML Experience (Q15)', fontsize=14)
  ax.set_ylabel('Role (Q5)', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(fig)

  st.write(''' ### Gender ###
  ''')
  st.write('''
  A striking 78.8% of the respondents are men. The analysis may overemphasize roles and skills more prevalent or preferred among men, while underrepresenting those associated with women. This can result in an incomplete understanding of the data industry's landscape. A role recommender system trained on gender-biased data might suggest career paths aligning with the majority gender's responses, thereby perpetuating existing gender disparities in the field. The result could unintentionally reinforce stereotypes by associating certain technical roles or skills predominantly with one gender, influencing perceptions and aspirations of future professionals.
         ''')


  # Calculate gender distribution per role
  gender_roles = df_no_S[1:].groupby(['Q5', 'Q2']).size().unstack().fillna(0)

  # Calculate the distribution of age
  gender = df_no_S.loc[1:, ['Q2']].value_counts().reset_index(name="Count")
  gender['Percent'] = gender.Count / gender.Count.sum() * 100
  gender.columns = ['Gender', 'Count', 'Percent']

  # Create a figure with two subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

  # Plot the gender distribution as a stacked bar chart on ax1
  gender_roles.plot(kind='bar', stacked=True, colormap='RdBu', ax=ax1)
  ax1.set_title('Gender Distribution Across Roles', fontsize=16)
  ax1.set_xlabel('Current Role (Q5)', fontsize=14)
  ax1.set_ylabel('Count of Respondents', fontsize=14)
  ax1.legend(title='Gender', loc='upper right')
  ax1.set_xticks(range(len(gender_roles.index))) 
  ax1.set_xticklabels(gender_roles.index, rotation=45, ha='right')

  # Plot the gedner distribution bar chart on ax2
  sns.barplot(x='Gender', y='Count', data=gender, ax=ax2)
  # Add percentages on top of the bars
  for i in range(len(gender)):
      ax2.text(i, gender['Count'].iloc[i] + 5, f"{gender['Percent'].iloc[i]:.1f}%", 
              ha='center', fontsize=10)

  # Customize the age plot
  ax2.set_title('Gender Distribution', fontsize=16)
  ax2.set_xlabel('Gender', fontsize=14)
  ax2.set_ylabel('Count', fontsize=14)

  # Display the combined plot in Streamlit
  st.pyplot(fig)

  st.write(''' ### Geographical Location ###
  ''')

  # Count the number of answers for each country (or any other column you'd like)
  country_distribution = df_no_S['Q3'].value_counts().reset_index(name='Count')
  country_distribution['Percent'] = country_distribution.Count / country_distribution.Count.sum() * 100
  country_distribution.columns = ['Country', 'Count', 'Percent']

  # Create the choropleth map
  fig = px.choropleth(country_distribution,
                        locations="Country",  # Column with country names or codes
                        locationmode="country names",  # Ensure country names are recognized
                        color="Count",  # Column to color the countries by
                        hover_name="Country",  # Add country name on hover
                        color_continuous_scale=px.colors.sequential.Blues,  # Color scale
                        labels={'Count': '# Responses'},  # Customize label in the legend
                        title="Distribution of Responses by Country",  # Map title
                        template='seaborn',  # Use seaborn style
                        height=600)  # Set the height of the map

  # Display the choropleth map in Streamlit
  st.plotly_chart(fig)

  # Filter top 10 countries
  country_respondents_top_10 = country_distribution.head(10)

  # Create a bar plot to display the number of respondents per country
  fig, ax = plt.subplots(figsize=(14, 8))
  sns.barplot(x='Count', y='Country', data=country_respondents_top_10, palette='Blues', ax=ax)

  # Add percentages on top of the bars
  for i in range(len(country_respondents_top_10)):
      ax.text(country_respondents_top_10['Count'].iloc[i] + 5, i, f"{country_respondents_top_10['Percent'].iloc[i]:.1f}%", 
              ha='left', fontsize=10)

  # Customize the plot
  plt.title('Number of Respondents per Country', fontsize=16)
  plt.xlabel('Number of Respondents', fontsize=14)
  plt.ylabel('Country', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(fig)

  st.write('''The survey shows a strong concentration of respondents from India, with 5851 participants, which significantly outnumbers other countries. 
  This indicates a high level of data-related activity or interest in India, possibly due to the country's growing tech sector and large pool of data science professionals.

  The survey also includes responses from a diverse range of countries, with representation from regions such as North America (USA), Europe (Germany, UK, Russia), and Asia (Japan, China), as well as countries in South America (Brazil) and Africa (Nigeria).
  This suggests that data science and related fields have a truly global presence, with professionals from different cultural and economic contexts participating in the survey.

  The presence of other countries like Brazil, Nigeria, and Russia indicate that there is increasing interest and growth in emerging markets as well.
  ''')

  st.write(''' ### Education ###
  ''')

  st.write('''Advanced education (Master’s and Doctoral degrees) is more common in senior roles like Data Scientist and Research Scientist, reflecting the specialized and research-oriented nature of these positions. 
  These roles often require higher-level expertise, and therefore, advanced education is a key factor.

  Entry-level roles like Data Analyst, Business Analyst, and Data Engineer show a broader range of educational backgrounds, indicating that these roles are more accessible. 
  People with Bachelor’s degrees, Master’s degrees, and even those with some college experience or no formal higher education are represented in these roles. This suggests that these positions offer pathways into the data industry for individuals with diverse educational backgrounds.

  Roles such as Data Analyst and Business Analyst are more flexible and can be seen as accessible entry points into the data field.
  This might also reflect the fact that these positions require strong analytical and problem-solving skills, which can often be gained through experience rather than formal advanced education.

  As roles become more specialized, such as Data Scientist or Research Scientist, advanced education becomes a more significant factor, demonstrating the importance of deep, specialized knowledge in these positions.
    ''')

  education_dist = df_no_S.loc[1:,['Q4']].value_counts().reset_index(name='Count')
  education_dist['Percent'] = education_dist['Count'] / education_dist['Count'].sum() * 100
  education_dist.columns = ['Education Level', 'Count', 'Percent']

  # Plot the bar plot with percentages
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.barplot(x='Count', y='Education Level', data=education_dist, palette='Blues', ax=ax)

  # Add percentages on top of the bars
  for i in range(len(education_dist)):
      ax.text(education_dist['Count'].iloc[i] + 5, i, f"{education_dist['Percent'].iloc[i]:.1f}%", 
              ha='left', va='center', fontsize=12)

  # Customize the plot
  ax.set_title('Education Level Distribution Among Respondents', fontsize=16)
  ax.set_xlabel('Number of Respondents', fontsize=14)
  ax.set_ylabel('Education Level', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(fig)


  # Group the data by role (Q5) and education level (Q4)
  role_education_distribution = df.iloc[1:].groupby(['Q5', 'Q4']).size().unstack().fillna(0)

  # Create the heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(role_education_distribution, annot=True, fmt='.0f', cmap='Blues', cbar=True)

  # Customize the plot
  plt.title('Role vs Education Level Distribution', fontsize=16)
  plt.xlabel('Education Level (Q4)', fontsize=14)
  plt.ylabel('Role (Q5)', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(plt)


  st.write(''' ### Activities ### ''')

  st.write(''' By examining the distribution of activities across different roles, we can conclude that Data Science is an interdisciplinary field.
   The top three activities encompass skills from business, programming, and statistics. 
   Analyzing and understanding data to influence products" is the most frequent activity, highlighting that data analysis for informed decision-making and product development is a central task for most professionals in the data industry. 
   Additionally, prototyping and experimenting with machine learning models are common tasks, underlining the significant role of machine learning in the field. ''')

  # Create the histogram for the 'Q23' column (activities)
  df_q = df_no_S.iloc[1:][[col for col in df.columns if 'Q23' in col]]

  # Collect activity counts
  activity_list = []
  for col in df_q:
      act_counts = df_q[col].value_counts()
      for activity, count in act_counts.items():
          activity_list.append({'Activity': activity, 'Count': count, 'Type': 'Use'})

  # Convert to DataFrame and aggregate duplicate activities
  activities = pd.DataFrame(activity_list)
  activities = activities.groupby('Activity', as_index=False).sum()
  activities = activities.sort_values(by='Count', ascending=False)

  # Calculate percentage
  activities["Percent"] = activities["Count"] / activities["Count"].sum() * 100

  # Create the plot
  fig, ax = plt.subplots(figsize=(10, 6))
  bars = sns.barplot(
      data=activities, 
      x='Count', 
      y='Activity',
      palette=sns.color_palette("coolwarm", as_cmap=False, n_colors=len(activities)),
      ax=ax
  )

  # Add percentage labels
  for index, value in enumerate(activities["Percent"]):
      ax.text(
          activities["Count"].iloc[index] + 2,  # Offset text slightly
          index,
          f"{value:.1f}%",
          va='center',
          fontsize=10
      )

  # Customize the plot
  ax.set_xlabel('Number of Respondents')
  ax.set_ylabel('Activity')
  ax.set_title('Distribution of Activities (Q23)')

  # Display in Streamlit
  st.pyplot(fig)




  # Extract the Q23-related columns and melt the dataframe
  column_indices_Q23 = [i for i, col in enumerate(df_no_S.columns) if 'Q23' in col]
  min_index = min(column_indices_Q23)
  max_index = max(column_indices_Q23)

  df_melt = df_no_S[1:].melt(id_vars='Q5', value_vars=df_no_S.iloc[:, min_index:max_index].T.index, var_name='Count', value_name='Value')

  # Remove roles we don't want
  data_melt = df_melt[~df_melt["Q5"].isin(['Student', 'Currently not employed', 'Other'])]

  # Group by Role and Activity
  data_meltTotal = data_melt.groupby(['Q5', 'Value']).count().reset_index()

  # Calculate total count for each role and percentage
  data_size = df_no_S.groupby(["Q5"]).size()
  data_meltTotal["Total"] = data_meltTotal["Q5"].map(data_size)
  data_meltTotal["Percent"] = round(data_meltTotal["Count"] / data_meltTotal["Total"] * 100, 2)

  # Create a button for each unique role
  unique_roles = data_meltTotal['Q5'].unique()

  # For each role, create a button and plot when clicked
  for role in unique_roles:
      # Display a button for each role
      if st.button(f"Show Activities for {role}"):
          # Create the plot only when the button is clicked
          fig, ax = plt.subplots(figsize=(12, 6))
          
          # Filter data for the selected role
          role_data = data_meltTotal[data_meltTotal['Q5'] == role]
          
          # Plot the bar chart for the role
          sns.barplot(data=role_data, x='Percent', y='Value', ax=ax, palette='coolwarm')
          
          # Customize the plot for better readability
          ax.set_title(f'Activities for {role}', fontsize=14)
          ax.set_xlabel('Percentage of Respondents', fontsize=12)
          ax.set_ylabel('Activity', fontsize=12)
          ax.tick_params(axis='y', labelsize=10)
          
          # Rotate labels to avoid overlap
          plt.setp(ax.get_yticklabels(), rotation=45, ha='right')

          # Show the plot in Streamlit
          st.pyplot(fig)





  st.write(''' ### Programming Language ###
  
  Python dominates with a large number of users. This is expected, as Python is a dominant language in data science, machine learning, and data analysis due to its versatility, ease of use, and extensive library support (like pandas, NumPy, and TensorFlow).
  SQL is the next most used language which makes sense because SQL is a foundational language for managing and querying relational databases. 
  Data professionals across various roles (Data Analysts, Data Engineers, etc.) frequently work with databases.
   ''')

 #Role vs Programming Language Used
  column_indices_Q7 = [i for i, col in enumerate(df.columns) if 'Q7' in col]

  # Extract the minimum and maximum indices
  min_index = min(column_indices_Q7)
  max_index = max(column_indices_Q7)

  df_melt = df_no_S[1:].melt(id_vars='Q5', value_vars=df.iloc[:,min_index:max_index].T.index,var_name='Count',value_name='Value')
  data_melt = df_melt[~df_melt["Q5"].isin(['Student','Currently not employed','Other'])]
  data_meltTotal = data_melt.groupby(['Q5','Value']).count().reset_index()

  data_size = df_no_S.groupby(["Q5"]).size()
  data_meltTotal["Total"] = data_meltTotal["Q5"].map(data_size)
  data_meltTotal["Percent"] = round(data_meltTotal["Count"] / data_meltTotal["Total"] * 100, 2)

  # Create the treemap plot
  fig = px.treemap(data_meltTotal,
                  path=['Q5', 'Value'],
                  height=700,
                  values='Percent',
                  title='Role vs Programming Language Used',
                  color_discrete_sequence=px.colors.sequential.RdBu)

  fig.update_traces(opacity=0.80, textfont_size=12,
                  texttemplate="%{label}<br><br>%{value:.2f}%"
                  )

  # Display the plot in Streamlit
  st.plotly_chart(fig)



  st.write(''' ### Annual compensation ###
  
  Higher-paying roles: Data Scientist, Machine Learning Engineer, and Software Engineer seem to be the roles most likely to command higher salaries, with many respondents in the 10,000 to 14,999 and 15,000 to $9,999 ranges.
  
  Lower-paying roles: Business Analysts and some of the more niche roles like DBA/Database Engineers appear to have a higher concentration of respondents in lower salary brackets (e.g., 1,000 to 1,999, 5,000 to 7,499).
 
  Entry and mid-level salaries: Many roles like Data Analyst, Product/Project Manager, and Research Scientist show a significant presence in the mid-range salary brackets (e.g., 5,000 to 7,499, 10,000 to 14,999), indicating that these positions might be filled by individuals with less experience or at lower levels in the career path compared to the more senior roles.
  ''')



  compensation_order = ['0-999','1000-1999','2000-2999','3000-3999','4000-4999','5000-7499','7500-9999', '10000-14999','15000-19999', '20000-24999','25000-29999','30000-39999','40000-49999','50000-59999','60000-69999','70000-79999','80000-89999','90000-99999','100000-124999','125000-149999','150000-199999','200000-249999','250000-299999','300000-500000','> 500000']

  anual_comp_dist = df_no_S.loc[1:,['Q24']].value_counts().reset_index(name='Count')
  anual_comp_dist['Percent'] = anual_comp_dist['Count'] / anual_comp_dist['Count'].sum() * 100
  anual_comp_dist.columns = ['Annual Compensation', 'Count', 'Percent']

  anual_comp_dist['Annual Compensation'] =anual_comp_dist['Annual Compensation'].replace({'\$': '', ',': ''}, regex=True)

  # Set the order of the compensation intervals
  anual_comp_dist['Annual Compensation'] = pd.Categorical(anual_comp_dist['Annual Compensation'], categories=compensation_order, ordered=True)

  # Sort the data by the defined order
  anual_comp_dist = anual_comp_dist.sort_values('Annual Compensation')


  # Plot the bar plot with percentages
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.barplot(x='Count', y='Annual Compensation', data=anual_comp_dist, palette='Blues', ax=ax)

  # Add percentages on top of the bars
  for i in range(len(anual_comp_dist)):
      ax.text(anual_comp_dist['Count'].iloc[i] + 50, i, f"{anual_comp_dist['Percent'].iloc[i]:.1f}%", 
              ha='left', va='center', fontsize=12)

  # Customize the plot
  ax.set_title('Annual Compensation Distribution Among Respondents', fontsize=16)
  ax.set_xlabel('Number of Respondents', fontsize=14)
  ax.set_ylabel('Annual Compensation', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(fig)




  # Apply the transformation to 'Q24' directly in the groupby operation
  role_anual_comp_dist = df_no_S.iloc[1:].copy()

  # Clean the 'Q24' column temporarily by replacing dollar signs and commas
  role_anual_comp_dist['Q24_clean'] = role_anual_comp_dist['Q24'].replace({'\$': '', ',': ''}, regex=True)

  # Perform the groupby operation using the cleaned version of 'Q24'
  role_anual_comp_dist = role_anual_comp_dist.groupby(['Q5', 'Q24_clean']).size().unstack().fillna(0)

  
  # Ensure the compensation categories are ordered in the heatmap
  role_anual_comp_dist = role_anual_comp_dist[compensation_order]

  # Create the heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(role_anual_comp_dist, annot=True, fmt='.0f', cmap='Blues', cbar=True)

  # Customize the plot
  plt.title('Role vs Annual Compensation Distribution', fontsize=16)
  plt.xlabel('Annual compensation', fontsize=14)

  plt.ylabel('Role', fontsize=14)

  # Display the plot in Streamlit
  st.pyplot(plt)


    

if page == pages[3] : 
  
  st.write("""### Training the Models ###

  This section is divided into 4 main parts:

  - Transform the data set
  - Model training
  - Features engineering on the selected model
  - Conclusions        
           
  **1. Transform the data set**  
             
  The columns generated by answers to Multiple-Selection Questions pose significant challenges. Each possible choice is represented as a separate column,
  making it difficult to analyze the frequency of each option, compare trends, or extract meaningful insights. 
  This structure also complicates machine learning tasks, as it results in a sparse and high-dimensional dataset, increasing the risk of overfitting. 
  Additionally, the lack of a compact representation makes it harder to apply standard algorithms that expect categorical or numerical inputs. 
                   
  """)  
  
  # Display original shape

  if 'original_df' not in st.session_state:
      st.session_state.original_df = df.copy()  # Store the original DataFrame

  # If 'df' is already in session_state, it might be modified, so reset it
  if 'df' not in st.session_state or st.button('Reset to Original DataFrame (20037, 355)'):
      st.session_state.df = st.session_state.original_df.copy()  # Reset to the original df (20037, 355)
      st.write(f"**Original state. Shape:** {st.session_state.df.shape}")

  st.write(""" **1.1. Data Cleaning**:  """)

  if st.button('Delete the questions from the dataframe (first row)'):
      st.session_state.df = st.session_state.df.drop(index=0).reset_index(drop=True)  # Modify df directly
      st.write(f"**New Shape after deleting the first row:** {st.session_state.df.shape}")

  if st.button('Delete the columns having only null values'):
      st.session_state.df = st.session_state.df.dropna(axis=1, how='all')  # Modify df directly
      st.write(f"**New Shape after deleting columns with null values:** {st.session_state.df.shape}")

  if st.button('Delete the duplicates rows'):
      st.session_state.df = st.session_state.df.drop_duplicates(inplace=False)  # Modify df directly
      st.write(f"**New Shape after deleting duplicates:** {st.session_state.df.shape}")

  if st.button('Remove Time from Start to Finish (seconds) column'):
      st.session_state.df.drop(['Time from Start to Finish (seconds)'], inplace=True, axis=1)
      st.write(f"**New Shape after removing 'Time from Start to Finish (seconds)' column:** {st.session_state.df.shape}")

  if st.button('Remove Student and Currently not employed data'):
      st.session_state.df = st.session_state.df[~st.session_state.df['Q5'].isin(['Student', 'Currently not employed'])]
      st.session_state.df.reset_index(drop=True, inplace=True)
      st.write(f"**New Shape after removing 'Student' and 'Currently not employed' data:** {st.session_state.df.shape}")

  if st.button('Drop rows where Q5 is null'):
      st.session_state.df = st.session_state.df.dropna(subset=['Q5'])
      st.session_state.df.reset_index(drop=True, inplace=True)
      st.write(f"**New Shape after dropping rows with null in Q5:** {st.session_state.df.shape}")

  if st.button('Drop columns Q30 and Q32 (above 90%  null values)'):
      st.session_state.df.drop(['Q30', 'Q32'], inplace=True, axis=1)
      st.session_state.df.reset_index(drop=True, inplace=True)
      st.write(f"**New Shape after dropping Q30 and Q32:** {st.session_state.df.shape}")

  st.write(f"**Current state of the data frame** {st.session_state.df.shape}")
  st.dataframe(st.session_state.df)
  st.write(""" **1.2. Data Encoding**: 
  - Encode the Target variable """)

  if 'X' in st.session_state:
    del st.session_state.X  # Delete old X

  if 'y' in st.session_state:
    del st.session_state.y  # Delete old y

    # Prepare the features (X) and target variable (y)
  if 'X' not in st.session_state and 'y' not in st.session_state:
    st.session_state.X = st.session_state.df.drop(columns=['Q5'])  # Replace 'Q5' with the actual target column
    st.session_state.y = st.session_state.df['Q5']  # Replace 'Q5' with the actual target column

  if 'y_encoded' not in st.session_state:
    st.session_state.y_encoded = None  # Initialize with None
  
  # Ensure session state variables are initialized
  if "X_train" not in st.session_state:
      st.session_state.X_train = None
      st.session_state.X_test = None
      st.session_state.y_train = None
      st.session_state.y_test = None

  # st.dataframe(st.session_state.X)
  # st.dataframe(st.session_state.y)


  if st.button("Apply Label Encoder"):
    # Apply Label Encoding before train_test_split
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(st.session_state.y)

    st.session_state.label_encoder = label_encoder  # Save encoder for decoding later 
    st.session_state.y_encoded =  y_encoded

    #Display original and encoded values
    st.write("### Original vs Encoded Values")
    y_encoded_df = pd.DataFrame({'Original': st.session_state.y , 'Encoded': st.session_state.y_encoded})
    st.dataframe(y_encoded_df)  # Display in Streamlit
    st.success("Label encoding applied successfully!")

  # # Display y_encoded outside the button (if it exists)
  # if st.session_state.y_encoded is not None:
  #     st.write("### Encoded Target (y_encoded)")
  #     st.dataframe(pd.DataFrame({'Encoded': st.session_state.y_encoded}))

   # Split the data into training and testing sets
  # st.write('''before split after I runt he encoder data is correctly encoded''')
  # st.dataframe(st.session_state.X)
  # st.dataframe(st.session_state.y_encoded)

  if st.session_state.y_encoded is None or len(st.session_state.y_encoded) == 0:
    st.warning("Target variable (y_encoded) is not encoded yet! Please apply label encoding first.")
  else:
     if st.button("Split Data"):
      st.write('''We will first split the Data Frame in test and train''')
      X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y_encoded, test_size=0.2, random_state=42)
      st.dataframe(X_train)
      st.dataframe(X_test)
      st.dataframe(y_train)
      st.dataframe(y_test)

      st.session_state.X_train = X_train
      st.session_state.X_test = X_test
      st.session_state.y_train = y_train
      st.session_state.y_test = y_test

      st.success("Data successfully split into train and test sets!")

      if st.session_state.X_train is not None:
        st.write("### Train and Test Sets")
        st.dataframe(st.session_state.X_train)
        st.dataframe(st.session_state.X_test)
        st.dataframe(st.session_state.y_train)
        st.dataframe(st.session_state.y_test)

  st.write("""
  Encoding after spliting the data into train and test        
   - Fill missing values for specific columns
   - Apply Ordinal Encoding to specific columns
   - Apply transformation for Multiple Selection Questions
   - Apply One-Hot Encoding to specific columns
   """)
  

  if (st.session_state.y_encoded is None or len(st.session_state.y_encoded) == 0) or \
    (st.session_state.X_train is None or len(st.session_state.X_train) == 0) or \
    (st.session_state.X_test is None or len(st.session_state.X_test) == 0) or \
    (st.session_state.y_train is None or len(st.session_state.y_train) == 0) or \
    (st.session_state.y_test is None or len(st.session_state.y_test) == 0):

    st.warning("Splitting not done! Please apply splitting first.")
  else:
    if st.button("Encode Data"):
      # st.success("Splitting done! Applying transformations...")
        # Apply transformations to X_train and X_test

      # 1. Fill missing values for specific columns
      columns_to_fill = ['Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q20', 'Q21', 'Q22', 'Q24', 'Q25', 'Q38']  
      for column in columns_to_fill:
          # Fill missing values in X_train
        if column in st.session_state.X_train.columns:
          st.session_state.X_train[column] = st.session_state.X_train[column].fillna(st.session_state.X_train[column].mode()[0])
            # Fill missing values in X_test
        if column in st.session_state.X_test.columns:
          st.session_state.X_test[column] = st.session_state.X_test[column].fillna(st.session_state.X_test[column].mode()[0])
      st.write('''Mode applied''')
      st.dataframe(st.session_state.X_train)
      st.dataframe(st.session_state.X_test)

      # 2. Apply Ordinal Encoding to specific columns
      columns_to_encode = ['Q1', 'Q4', 'Q6', 'Q13', 'Q15', 'Q24', 'Q20', 'Q21', 'Q25']
      ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Fit and transform on X_train, then transform on X_test
      st.session_state.X_train[columns_to_encode] = ordinal_encoder.fit_transform(st.session_state.X_train[columns_to_encode])
      st.session_state.X_test[columns_to_encode] = ordinal_encoder.transform(st.session_state.X_test[columns_to_encode])
      st.write('''Ordinal encoding applied''')
      st.dataframe(st.session_state.X_train)
      st.dataframe(st.session_state.X_test)

        # 3. Apply transformation for Multiple Selection Questions (like Q7_Part_1)
      part_columns =  ['Q7_Part_1', 'Q7_Part_2', 'Q7_Part_3', 'Q7_Part_4', 'Q7_Part_5', 'Q7_Part_6', 'Q7_Part_7', 'Q7_Part_8', 'Q7_Part_9', 'Q7_Part_10', 'Q7_Part_11', 'Q7_OTHER', 
                        'Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5', 'Q9_Part_6', 'Q9_Part_7', 'Q9_Part_8', 'Q9_Part_9', 'Q9_Part_10', 'Q9_OTHER', 
                        'Q10_Part_1', 'Q10_Part_2', 'Q10_Part_3', 'Q10_Part_4', 'Q10_Part_5', 'Q10_Part_6', 'Q10_Part_7', 'Q10_Part_8', 'Q10_Part_9', 'Q10_Part_10', 'Q10_Part_11', 'Q10_Part_12', 'Q10_OTHER', 
                        'Q12_Part_1', 'Q12_Part_2', 'Q12_OTHER', 
                        'Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6', 'Q14_Part_7', 'Q14_Part_8', 'Q14_Part_9', 'Q14_Part_10', 'Q14_OTHER',
                        'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7', 'Q16_Part_8', 'Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12', 'Q16_Part_13', 'Q16_Part_14', 'Q16_OTHER', 
                        'Q17_Part_1', 'Q17_Part_2', 'Q17_Part_3', 'Q17_Part_4', 'Q17_Part_5', 'Q17_Part_6', 'Q17_Part_7', 'Q17_Part_8', 'Q17_Part_9', 'Q17_Part_10', 'Q17_OTHER', 
                        'Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5', 'Q18_OTHER', 
                        'Q19_Part_1', 'Q19_Part_2', 'Q19_Part_3', 'Q19_Part_4', 'Q19_OTHER',
                        'Q23_Part_1', 'Q23_Part_2', 'Q23_Part_3', 'Q23_Part_4', 'Q23_Part_5', 'Q23_Part_6', 'Q23_Part_7', 'Q23_OTHER', 
                        'Q26_A_Part_1', 'Q26_A_Part_2', 'Q26_A_Part_3', 'Q26_A_Part_4', 'Q26_A_Part_5', 'Q26_A_Part_6', 'Q26_A_Part_7', 'Q26_A_Part_8', 'Q26_A_Part_9', 'Q26_A_Part_10', 'Q26_A_OTHER', 
                        'Q27_A_Part_1', 'Q27_A_Part_2', 'Q27_A_Part_3', 'Q27_A_Part_4', 'Q27_A_Part_5', 'Q27_A_Part_6', 'Q27_A_Part_7', 'Q27_A_Part_8', 'Q27_A_Part_9', 'Q27_A_Part_10', 'Q27_A_Part_11', 'Q27_A_OTHER', 
                        'Q28_A_Part_1', 'Q28_A_Part_2', 'Q28_A_Part_3', 'Q28_A_Part_4', 'Q28_A_Part_5', 'Q28_A_Part_6', 'Q28_A_Part_7', 'Q28_A_Part_8', 'Q28_A_Part_9', 'Q28_A_Part_10', 'Q28_A_OTHER', 
                        'Q29_A_Part_1', 'Q29_A_Part_2', 'Q29_A_Part_3', 'Q29_A_Part_4', 'Q29_A_Part_5', 'Q29_A_Part_6', 'Q29_A_Part_7', 'Q29_A_Part_8', 'Q29_A_Part_9', 'Q29_A_Part_10', 'Q29_A_Part_11', 'Q29_A_Part_12', 'Q29_A_Part_13', 'Q29_A_Part_14', 'Q29_A_Part_15', 'Q29_A_Part_16', 'Q29_A_OTHER', 
                        'Q31_A_Part_1', 'Q31_A_Part_2', 'Q31_A_Part_3', 'Q31_A_Part_4', 'Q31_A_Part_5', 'Q31_A_Part_6', 'Q31_A_Part_7', 'Q31_A_Part_8', 'Q31_A_Part_9', 'Q31_A_Part_10', 'Q31_A_Part_11', 'Q31_A_Part_12', 'Q31_A_Part_13', 'Q31_A_OTHER', 
                        'Q33_A_Part_1', 'Q33_A_Part_2', 'Q33_A_Part_3', 'Q33_A_Part_4', 'Q33_A_Part_5', 'Q33_A_Part_6', 'Q33_A_Part_7', 'Q33_A_OTHER', 
                        'Q34_A_Part_1', 'Q34_A_Part_2', 'Q34_A_Part_3', 'Q34_A_Part_4', 'Q34_A_Part_5', 'Q34_A_Part_6', 'Q34_A_Part_7', 'Q34_A_Part_8', 'Q34_A_Part_9', 'Q34_A_Part_10', 'Q34_A_Part_11', 'Q34_A_OTHER', 
                        'Q35_A_Part_1', 'Q35_A_Part_2', 'Q35_A_Part_3', 'Q35_A_Part_4', 'Q35_A_Part_5', 'Q35_A_Part_6', 'Q35_A_Part_7', 'Q35_A_Part_8', 'Q35_A_Part_9', 'Q35_A_Part_10', 'Q35_A_OTHER', 
                        'Q36_Part_1', 'Q36_Part_2', 'Q36_Part_3', 'Q36_Part_4', 'Q36_Part_5', 'Q36_Part_6', 'Q36_Part_7', 'Q36_Part_8', 'Q36_Part_9', 'Q36_OTHER', 
                        'Q37_Part_1', 'Q37_Part_2', 'Q37_Part_3', 'Q37_Part_4', 'Q37_Part_5', 'Q37_Part_6', 'Q37_Part_7', 'Q37_Part_8', 'Q37_Part_9', 'Q37_Part_10', 'Q37_OTHER', 
                        'Q39_Part_1', 'Q39_Part_2', 'Q39_Part_3', 'Q39_Part_4', 'Q39_Part_5', 'Q39_Part_6', 'Q39_Part_7', 'Q39_Part_8', 'Q39_Part_9', 'Q39_Part_10', 'Q39_OTHER', 
                        'Q26_B_Part_1', 'Q26_B_Part_2', 'Q26_B_Part_3', 'Q26_B_Part_4', 'Q26_B_Part_5', 'Q26_B_Part_6', 'Q26_B_Part_7', 'Q26_B_Part_8', 'Q26_B_Part_9', 'Q26_B_Part_10', 'Q26_B_OTHER', 
                        'Q27_B_Part_1', 'Q27_B_Part_2', 'Q27_B_Part_3', 'Q27_B_Part_4', 'Q27_B_Part_5', 'Q27_B_Part_6', 'Q27_B_Part_7', 'Q27_B_Part_8', 'Q27_B_Part_9', 'Q27_B_Part_10', 'Q27_B_OTHER', 
                        'Q28_B_Part_1', 'Q28_B_Part_2', 'Q28_B_Part_3', 'Q28_B_Part_4', 'Q28_B_Part_5', 'Q28_B_Part_6', 'Q28_B_Part_7', 'Q28_B_Part_8', 'Q28_B_Part_9', 'Q28_B_OTHER', 
                        'Q29_B_Part_1', 'Q29_B_Part_2', 'Q29_B_Part_3', 'Q29_B_Part_4', 'Q29_B_Part_5', 'Q29_B_Part_6', 'Q29_B_Part_7', 'Q29_B_Part_8', 'Q29_B_Part_9', 'Q29_B_Part_10', 'Q29_B_Part_11', 'Q29_B_Part_12', 'Q29_B_Part_13', 'Q29_B_Part_14', 'Q29_B_Part_15', 'Q29_B_Part_16', 'Q29_B_OTHER', 
                        'Q31_B_Part_1', 'Q31_B_Part_2', 'Q31_B_Part_3', 'Q31_B_Part_4', 'Q31_B_Part_5', 'Q31_B_Part_6', 'Q31_B_Part_7', 'Q31_B_Part_8', 'Q31_B_Part_9', 'Q31_B_Part_10', 'Q31_B_Part_11', 'Q31_B_Part_12', 'Q31_B_Part_13', 'Q31_B_OTHER', 
                        'Q33_B_Part_1', 'Q33_B_Part_2', 'Q33_B_Part_3', 'Q33_B_Part_4', 'Q33_B_Part_5', 'Q33_B_Part_6', 'Q33_B_OTHER', 
                        'Q34_B_Part_1', 'Q34_B_Part_2', 'Q34_B_Part_3', 'Q34_B_Part_4', 'Q34_B_Part_5', 'Q34_B_Part_6', 'Q34_B_Part_7','Q34_B_Part_8', 'Q34_B_Part_9', 'Q34_B_Part_10', 'Q34_B_OTHER', 
                        'Q35_B_Part_1', 'Q35_B_Part_2', 'Q35_B_Part_3', 'Q35_B_Part_4', 'Q35_B_Part_5', 'Q35_B_Part_6', 'Q35_B_Part_7', 'Q35_B_Part_8', 'Q35_B_Part_9', 'Q35_B_OTHER'
                      ]
      for column in part_columns:
        if column in st.session_state.X_train.columns:
          st.session_state.X_train[column] = st.session_state.X_train[column].apply(lambda x: 1 if pd.notnull(x) else 0)
        if column in st.session_state.X_test.columns:
          st.session_state.X_test[column] = st.session_state.X_test[column].apply(lambda x: 1 if pd.notnull(x) else 0)
      st.write('''0 1 encoding applied''')
      st.dataframe(st.session_state.X_train)
      st.dataframe(st.session_state.X_test)

        # 4. Apply One-Hot Encoding to specific columns
      columns_to_encode = ['Q2', 'Q3', 'Q8', 'Q11', 'Q22', 'Q38']
        
        # Apply One-Hot Encoding to both X_train and X_test
      st.session_state.X_train = pd.get_dummies(st.session_state.X_train, columns=columns_to_encode, drop_first=False)
      st.session_state.X_test = pd.get_dummies(st.session_state.X_test, columns=columns_to_encode, drop_first=False)
        
        # Map boolean columns to 0 and 1 in both datasets
      st.session_state.X_train = st.session_state.X_train.apply(lambda x: x.map({True: 1, False: 0}) if x.dtype == 'bool' else x)
      st.session_state.X_test = st.session_state.X_test.apply(lambda x: x.map({True: 1, False: 0}) if x.dtype == 'bool' else x)
      st.write('''One-Hot Encoding applied''')
      st.dataframe(st.session_state.X_train)
      st.dataframe(st.session_state.X_test)




  if (st.session_state.y_encoded is None or len(st.session_state.y_encoded) == 0) or \
    (st.session_state.X_train is None or len(st.session_state.X_train) == 0) or \
    (st.session_state.X_test is None or len(st.session_state.X_test) == 0):

    st.warning("Encoding not done! Please apply splitting first.")
  else:
    if st.button('Show Transformed DataFrames'):
      st.write(f"X_train: {st.session_state.X_train.shape}")
      st.dataframe(st.session_state.X_train.head(5))
      st.write(f"X_test:  {st.session_state.X_test.shape}")
      st.dataframe(st.session_state.X_test.head(5))




  st.write(""" **1.3. Training the Data**:  """)
  st.write(""" In this part we present the tests carried out using various classification models.
    - Logistic regression
    - K nearest neighbors
    - Random Forest
    - Decision Tree
    - SVC (Support Vector Classifier)     
   """)  

  if (st.session_state.y_encoded is None or len(st.session_state.y_encoded) == 0) or \
    (st.session_state.X_train is None or len(st.session_state.X_train) == 0) or \
    (st.session_state.X_test is None or len(st.session_state.X_test) == 0):

    st.warning("Data is not prepared. Please prepare the data!")
  else:
    # Dropdown menu for model selection
    model_selection = st.selectbox( 'Select Model',('Logistic Regression','K nearest neighbors', 'Random Forest', 'Decision Tree','Support Vector Machine','Gradient Boost') )
  # Button to run the selected model
    if st.button('Run Model'):
         # Initialize the model based on the selected model
      if model_selection == 'Logistic Regression':
        model = LogisticRegression()
      elif model_selection == 'K nearest neighbors':
        model = KNeighborsClassifier()
      elif model_selection == 'Random Forest':
        model = RandomForestClassifier()
      elif model_selection == 'Decision Tree':
        model = DecisionTreeClassifier()
      elif model_selection == 'Support Vector Machine':
        model = svm.SVC()
      elif model_selection == 'Gradient Boost':
        model = GradientBoostingClassifier()
    # st.write(f"{model_selection}")
    # st.dataframe(st.session_state.X_train)
    # st.dataframe(st.session_state.X_test)
    # st.dataframe(st.session_state.y_train)
    # st.dataframe(st.session_state.y_test)


            # Train the model
      model.fit(st.session_state.X_train, st.session_state.y_train)

            # Make predictions
      y_pred = model.predict(st.session_state.X_test)

            # Evaluate the model
      accuracy = accuracy_score(st.session_state.y_test, y_pred)

            # Display the results
      st.write(f'Model: {model_selection}')
      st.write(f'Accuracy: {accuracy * 100:.2f}%')

            # Generate the confusion matrix
      cm_df = pd.crosstab(st.session_state.y_test, y_pred, rownames=['Real Class'], colnames=['Predicted Class'])

            # Display the confusion matrix as a table
      st.write("### Confusion Matrix")
      st.write(cm_df)

              # Display classification report
      report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
      report_df = pd.DataFrame(report).transpose()
      st.write("**Classification Report:**")
      st.dataframe(report_df)

              # Visualization: Confusion Matrix Heatmap
      fig, ax = plt.subplots(figsize=(8,6))
      sns.heatmap(cm_df, annot=True, fmt="d", cmap='Blues', xticklabels=cm_df.columns, yticklabels=cm_df.index)
      plt.title('Confusion Matrix')
      plt.xlabel('Predicted Class')
      plt.ylabel('True Class')
      st.pyplot(fig)


    #Display a table with the models accuracy
    st.write('''Model Accuracy:''')
    accuracy_table = {
      'Name': ['Logistic Regression','K nearest neighbors', 'Random Forest', 'Decision Tree','Support Vector Machine','Gradient Boost'],
      'Accuracy': [40.26 ,24.5, 39.26, 22.40, 39.86, 41.19]
    }
    accuracy_table = pd.DataFrame(accuracy_table)
    styled_accuracy = accuracy_table.style.background_gradient(axis=None, cmap='Blues', subset=['Accuracy'])
    st.dataframe(styled_accuracy)

  #  - The Gradient Boosting: Has the highest accuracy 41.19, which is better than the others.
  # However, accuracy alone is not enough to decide which model is the best, especially when dealing with imbalanced data.

  # - Logistic Regression: Overall low precision and recall, especially for classes like 0, 1, and 3, indicating it struggles with some classes.
  # Precision and recall for class 4 (likely a dominant class) are much better, reflecting that the model is doing better at predicting this class. Also it does not converge even with a higher number of iterations.
           
  # - SVC: Similar to Logistic Regression in terms of performance: low precision and recall for many classes, but better for 4 and 9.
  # Precision and recall for class 4 are also high, but overall, the model is not performing well across all classes, as shown by the low macro average scores.
           
  # - Random Forest: Like SVC, low precision and recall for many classes, especially 0, 1, 3, and 7.
  # However, class 4 has good recall, suggesting that the model can identify it well.
  # Accuracy is close to SVC and Logistic Regression, but macro and weighted averages suggest that performance across all classes isn’t balanced.

    st.write('''            
    - Gradient Boosting:
    Best overall performance based on accuracy and the F1-score.
    The F1-score for 4 is the highest among all models, and class 9 also has a good F1-score.
    Macro average is balanced at 0.30, which is higher than the others, showing that the model performs better across a wider range of classes.
    Precision and recall for class 9 are also strong, giving Gradient Boosting an edge.
            
    The performance is clearly lacking, so we decided to experiment with different techniques to enhance it. Before deciding on a model we tried Oversampling and PCA .
      ''')

    st.write(""" **A.** To address the class imbalance, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** to improve the class distribution.""")
    if st.button("Show Class Distribution Before SMOTE"):
      st.write("### Class Distribution Before Applying SMOTE")
      fig_before, ax_before = plt.subplots()
      sns.countplot(x=st.session_state.y_train, ax=ax_before, palette="coolwarm")
      ax_before.set_title("Class Distribution Before SMOTE")
      st.pyplot(fig_before)


    # Button to apply SMOTE
    if st.button("Apply SMOTE and Show Class Distribution After"):
        # Apply SMOTE and store in session state
        smote = SMOTE(random_state=42)
        X_train_OS, y_train_OS = smote.fit_resample(st.session_state.X_train, st.session_state.y_train)

        # Display class distribution after SMOTE
        fig_after, ax_after = plt.subplots()
        sns.countplot(x=y_train_OS, ax=ax_after, palette="coolwarm")
        ax_after.set_title("Class Distribution After SMOTE")
        st.pyplot(fig_after)

    st.write("We applied the machine learning algorithm listed above using the new sample but we did not noticed any improvement.")

    st.write(""" **B.** To reduce dimensionality, we applied **Principal Component Analysis (PCA)** to simplify the dataset while preserving its variance.""")

    if st.button("Apply Principal Component Analysis (PCA)"):
     # Apply PCA
      pca = PCA(n_components=50)  
      X_train_pca = pca.fit_transform(st.session_state.X_train)
      X_test_pca = pca.transform(st.session_state.X_test)

            # Explained variance ratio
      explained_variance_ratio = pca.explained_variance_ratio_

      # Cumulative explained variance
      cumulative_variance = np.cumsum(explained_variance_ratio)

      # Plot explained variance vs number of components
      plt.figure(figsize=(10, 6))
      plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', color='b', label='Explained Variance')
      plt.xlabel('Number of Components')
      plt.ylabel('Explained Variance Ratio')
      plt.title('Explained Variance vs. Number of Components')
      plt.grid(True)
      plt.legend(loc='best')
      st.pyplot(plt)

      # Plot cumulative explained variance vs number of components
      plt.figure(figsize=(10, 6))
      plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='g', label='Cumulative Explained Variance')
      plt.axhline(y=0.90, color='r', linestyle='--', label="90% Variance Threshold")  # 90% explained variance threshold
      plt.xlabel('Number of Components')
      plt.ylabel('Cumulative Explained Variance')
      plt.title('Cumulative Explained Variance vs. Number of Components')
      plt.grid(True)
      plt.legend(loc='best')
      st.pyplot(plt)

    st.write("""The variance per component is quite small, and it decreases rapidly, with the variance being distributed very thinly. As a result, PCA struggles to capture any meaningful variance from the data.
          Although we applied machine learning algorithms for the exercise, the performance did not improve and, in fact, degraded.
          While the exercise was valuable from an educational perspective, it does not provide significant insights for further modeling.
            """)

#   #     # Confirm success
#   #     st.success("SMOTE applied successfully! The dataset is now balanced.")

#   #   # Dropdown menu for model selection
#   # model_selection = st.selectbox(
#   #     'Select Model after Oversampling',
#   #      ('Logistic Regression', 'K nearest neighbors', 'Random Forest', 'Decision Tree', 'Support Vector Machine')
#   # )

#   # # Check if SMOTE has been applied before training
#   # if 'X_train_AO' in st.session_state and 'y_train_AO' in st.session_state:
#   #     # Button to run the selected model
#   #     if st.button('Run Model after Oversampling'):
#   #         # Select the model
#   #         if model_selection == 'Logistic Regression':
#   #             model = LogisticRegression()
#   #         elif model_selection == 'K nearest neighbors':
#   #             model = KNeighborsClassifier()
#   #         elif model_selection == 'Random Forest':
#   #             model = RandomForestClassifier()
#   #         elif model_selection == 'Decision Tree':
#   #             model = DecisionTreeClassifier()
#   #         elif model_selection == 'Support Vector Machine':
#   #             model = svm.SVC()

#   #         # Train the model
#   #         model.fit(st.session_state.X_train_AO, st.session_state.y_train_AO)

#   #         # Make predictions
#   #         y_pred = model.predict(X_test)

#   #         # Evaluate the model
#   #         accuracy = accuracy_score(y_test, y_pred)

#   #         # Display the results
#   #         st.write(f'**Model:** {model_selection}')
#   #         st.write(f'**Accuracy:** {accuracy * 100:.2f}%')

#   #         # Generate the confusion matrix
#   #         cm_df = pd.crosstab(y_test, y_pred, rownames=['Real Class'], colnames=['Predicted Class'])

#   #         # Display the confusion matrix as a table
#   #         st.write("### Confusion Matrix")
#   #         st.write(cm_df)

#   #         # Display classification report
#   #         report = classification_report(y_test, y_pred, output_dict=True)
#   #         report_df = pd.DataFrame(report).transpose()
#   #         st.write("**Classification Report:**")
#   #         st.dataframe(report_df)

#   #         # Visualization: Confusion Matrix Heatmap
#   #         fig, ax = plt.subplots(figsize=(8, 6))
#   #         sns.heatmap(cm_df, annot=True, fmt="d", cmap='Blues', xticklabels=cm_df.columns, yticklabels=cm_df.index)
#   #         plt.title('Confusion Matrix')
#   #         plt.xlabel('Predicted Class')
#   #         plt.ylabel('True Class')
#   #         st.pyplot(fig)
#   # else:
#   #     st.warning("Please apply SMOTE before running a model.")



    st.write('''
    Best Model: **Gradient Boosting**

    Higher accuracy: 41.19%, the best among the models.
    Better F1-scores for multiple classes, especially 4 and 9, where precision and recall are also higher.
    It balances precision and recall better, meaning it is more consistent across different classes.
    Macro average F1-score is higher than the others, indicating better overall performance across all classes.
    Conclusion:
    Gradient Boosting is the best model here, as it provides the highest accuracy and balances precision and recall better across the various classes, especially for the important ones. 
            ''')

    st.write('''**1.4. Features engineering on the selected model**''')
    st.write('''Steps taken to improve the model and their effects.''')
    test_table = {
      'Name': ['Oversampling', 'Undersampling', 'PCA', 'Feature Importance Selection','Hyperparameter adjustment: Learning_rate = 0.01'],
      'Accuracy': [0.30, 0.27, 0.37, 0.40, 0.38]
    }

    test_table = pd.DataFrame(test_table)

    # Apply a color gradient to the numeric columns (darker for higher values)
    styled_test = test_table.style.background_gradient(axis=None, cmap='Blues', subset=['Accuracy'])

    # Display the styled table
    st.dataframe(styled_test)

  st.write('''**1.5. Conclusion**''')
  st.write(''' The 'raw' model achieves the highest accuracy score, but the overall accuracy remains low. Unfortunately, these models will not be sufficient for an effective recommendation syste''')
  st.write(""" We performed a test with some sample data selected from X_test:""")
  # index  = 10
  # # st.write(type(st.session_state.y_test))
  # X_sample = st.session_state.X_test.iloc[[index]]
  # y_sample = st.session_state.y_test[index]
  # st.write(X_sample)
  # st.write(y_sample)
  # # st.dataframe(st.session_state.X_test)


  st.write('''**1.6. Test**''')
  
  if (st.session_state.y_encoded is None or len(st.session_state.y_encoded) == 0) or \
    (st.session_state.X_train is None or len(st.session_state.X_train) == 0) or \
    (st.session_state.X_test is None or len(st.session_state.X_test) == 0):

    st.warning("Data is not prepared. Please prepare the data!")
  else:
    # gb_model = GradientBoostingClassifier()
    # gb_model.fit(st.session_state.X_train, st.session_state.y_train)
  # Dropdown for selecting an index
    index = st.selectbox("Select an index", list(range(0, 10)))

  # Button to make a prediction
    if st.button("Predict"):
      Initialize the Gradient Boosting model
      gb_model = GradientBoostingClassifier()
      gb_model.fit(st.session_state.X_train, st.session_state.y_train)
      X_sample = st.session_state.X_test.iloc[[index]]
      y_sample = st.session_state.y_test[index] 
      y_pred_sample = gb_model.predict(X_sample)
        
      st.write(f"**Predicted Class:** {y_pred_sample[0]}")
      st.write(f"**Actual Class:** {y_sample}")




