import pandas as pd
import streamlit as st
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@st.cache
def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.drop('Unnamed: 0', axis=1)  # Assuming you want to drop this column as in your initial code
    return data

data_path=os.path.join('artifacts','data_full_features.csv')

data=load_data(data_path)
#print(data.head())

dataExploration = st.container()

with dataExploration:
  st.title('Philadelphia Crime Analysis')
  st.header('Dataset: Philadelphia Crime')
  st.markdown('I found this dataset at... https://data.phila.gov/visualizations/crime-incidents')
  st.markdown('**It is a "Philadelphia Crime Incidents" dataset from the year 2006 - Present**')
  st.markdown('****Note: I have taken the data upto October 2023****')
  st.text('Below is the sample DataFrame')


col1, col2 = st.columns(2)
with col1:
    st.markdown("**Total Crimes**")
    st.write(data.shape[0])

with col2:
   st.markdown("**Total Crimes Types**")
   st.write(data['crime_type'].nunique())
  

plotContainer = st.container()
with plotContainer:
    st.header('Crime Type Analysis')
    st.subheader("Let's find out which type of crime most reported?")
    crime_counts = data['crime_type'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=data, y='crime_type', color='#ff5252', order=crime_counts.index)
    plt.title('Crime Type Count',fontsize=14, fontweight='bold')
    plt.xlabel('Count',fontsize=14)
    plt.ylabel('Crime Type',fontsize=14)

    # Iterate through the list of axes' patches
    for p in ax.patches:
        width = p.get_width()    # Get the width of the bar
        ax.text(width + 3,       # Set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Get the vertical position
                f'{int(width)}',  # The label text
                va='center')  # Center alignment

    st.pyplot(plt)
    st.markdown('**Insights**')
    st.markdown(f"*1. Theft is the most reported crime type : 404177*")


def categorize_time(hour):
    if 5 <= hour <= 7:
        return "Early Morning"
    elif 8 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 15:
        return "Afternoon"
    elif 16 <= hour <= 19:
        return "Evening"
    elif 20 <= hour <= 23:
        return "Late Evening"
    else:
        return "Night"

# Apply the function to create a new column
data['time_of_day'] = data['Hour'].apply(categorize_time)
time_categories = {
    'Time of Day': ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Evening', 'Night'],
    'Hour Range': ['5 AM - 7 AM', '8 AM - 11 AM', '12 PM - 3 PM', '4 PM - 7 PM', '8 PM - 11 PM', '12 AM - 4 AM']
}
time_df = pd.DataFrame(time_categories)

plotContainer2 = st.container()
with plotContainer2:
    st.header('Time of Day Crime Analysis')
    st.subheader("Let's find out what time of day are most crime reported?")
    st.markdown('****To do this, I have converted the hours column to the following:****')
    st.table(time_df)
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=data, y='time_of_day', color='#ff5252', order=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Evening', 'Night'])
    plt.title('Time of Day VS Crime Count',fontsize=14, fontweight='bold')
    plt.xlabel('Count',fontsize=14)
    plt.ylabel('Time of Day',fontsize=14)

    # Iterate through the list of axes' patches to add text (counts)
    for p in ax.patches:
        width = p.get_width()  # Get the width of the bar, which represents the count
        ax.text(width + 3,  # Position the text slightly to the right of the bar end
                p.get_y() + p.get_height() / 2,  # Set the vertical position to the middle of the bar
                f'{int(width)}',  # The label text with the count
                va='center')  # Center the text vertically

    #plt.show()
    st.pyplot(plt)
    st.markdown('**Insights**')
    st.markdown(f'*1. Between 4 PM - 7 PM (Evening) the most number of crime are reported : 673690*')
    st.markdown(f'*2. Between 5 AM - 7 AM (Early Morning) the least number of crime are reported : 108751*')
    st.markdown(f'*3. Even though many people think night (between 12 AM - 4 AM) is the most dangerous time, the data tells us fewer crimes are reported then, with 389,754 incidents at night. This raises a question when do severe crimes occur?*')
    
# Given crime_severity_mapping dictionary
crime_severity_mapping = {
    'Robbery Firearm': 'Severe',
    'Other Assaults': 'Moderate',
    'Thefts': 'Minor',
    'Vandalism/Criminal Mischief': 'Moderate',
    'Burglary Non-Residential': 'Moderate',
    'Motor Vehicle Theft': 'Moderate',
    'Aggravated Assault No Firearm': 'Severe',
    'Robbery No Firearm': 'Moderate',
    'Weapon Violations': 'Moderate',
    'Fraud': 'Minor',
    'Burglary Residential': 'Moderate',
    'Liquor Law Violations': 'Minor',
    'Receiving Stolen Property': 'Minor',
    'All Other Offenses': 'Minor',
    'Aggravated Assault Firearm': 'Severe',
    'Theft from Vehicle': 'Minor',
    'Rape': 'Severe',
    'Other Sex Offenses (Not Commercialized)': 'Moderate',
    'Narcotic / Drug Law Violations': 'Moderate',
    'Disorderly Conduct': 'Minor',
    'DRIVING UNDER THE INFLUENCE': 'Moderate',
    'Arson': 'Severe',
    'Embezzlement': 'Minor',
    'Forgery and Counterfeiting': 'Minor',
    'Homicides': 'Severe',
    'Offenses Against Family and Children': 'Moderate',
    'Prostitution and Commercialized Vice': 'Moderate',
    'Public Drunkenness': 'Minor',
    'Vagrancy/Loitering': 'Minor',
    'Gambling Violations': 'Minor'
}

crime_severity_df = pd.DataFrame(list(crime_severity_mapping.items()), columns=['Crime Type', 'Severity'])

data['degree_of_crime'] = data['crime_type'].map(crime_severity_mapping)


data['degree_of_crime'].value_counts()

crime_degree_colors = {
    'Minor': '#ffbaba',
    'Moderate': '#ff5252',
    'Severe': '#a70000'
}



plotContainer3 = st.container()
with plotContainer3:
    st.header('Time of Day Crime Severity Analysis')
    st.subheader("Let's find out severity of crimes reported during different times of day")
    st.markdown('****To do this, I have converted the hours column as shown before.****')
    st.markdown('****The severity has been mapped as following:****')
    st.table(crime_severity_df)

    degree_of_crime_counts=dict(data['degree_of_crime'].value_counts())

# Data to plot
    labels = degree_of_crime_counts.keys()
    sizes = degree_of_crime_counts.values()
    colors = ['#ffbaba', '#ff5252', '#a70000']


    # Plot
    plt.figure(figsize=(8, 8))
    plt.pie(sizes,  labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=140)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Crime Severity Distribution')
    st.pyplot(plt)
    grouped_counts = data.groupby(['time_of_day', 'degree_of_crime']).size().reset_index(name='counts')

# Plotting the bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=grouped_counts, y='time_of_day', x='counts', hue='degree_of_crime',
                    order=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Evening', 'Night'],
                    palette=crime_degree_colors)

    plt.title('Crime Count by Time of Day and Severity',fontsize=14, fontweight='bold')
    plt.xlabel('Count',fontsize=14)
    plt.ylabel('Time of Day',fontsize=14)
    plt.legend(title='Degree of Crime', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=10)
    plt.tight_layout()

    # Now, iterate over the grouped_counts DataFrame to place text annotations
    for index, row in grouped_counts.iterrows():
        # Find the position for the text
        y_position = ax.get_yticklabels().index(ax.get_yticklabels()[[label.get_text() for label in ax.get_yticklabels()].index(row['time_of_day'])])
        hue_offset = {'Minor': -0.25, 'Moderate': 0, 'Severe': 0.3}  # Adjust these offsets as needed for your data
        ax.text(row['counts'] + 3, y_position + hue_offset[row['degree_of_crime']],
                str(row['counts']), color='black', ha="left", va="center")

    st.pyplot(plt)
    st.markdown('**Insights**')
    st.markdown(f'*1. As the day progresses the number of severe crimes reported increases*')
    st.markdown(f'*2. Between 8 PM - 11 PM (Late Evening) the most number of severe crimes are reported : 50502*')
    st.markdown(f'*3. Even though crimes reported in evening (Between 4 PM - 7 PM ) are the highest, 48% are minor crimes and 44% are moderate crimes*')


timeseries=data.copy()
# Assuming your data is loaded into a DataFrame named `df`
# First, ensure that your date and time columns are integers
timeseries['Year'] = timeseries['Year'].astype(int)
timeseries['Month'] = timeseries['Month'].astype(int)
timeseries['Day'] = timeseries['Day'].astype(int)
timeseries['Hour'] = timeseries['Hour'].astype(int)
timeseries['Minute'] = timeseries['Minute'].astype(int)

# Create a new datetime column by combining Year, Month, Day, Hour, and Minute
timeseries['Datetime'] = pd.to_datetime(timeseries[['Year', 'Month', 'Day', 'Hour', 'Minute']])
timeseries=timeseries.drop(['Year','Month','Day','Hour','Minute'],axis=1)

crime_count_per_day = timeseries.groupby(timeseries['Datetime'].dt.date).size()

#crime_count_per_day.plot(color='#ff5252')

def plot_crime_data(crime_count_per_day):
    # Your existing code
    plt.clf()
    crime_count_per_day.index = pd.to_datetime(crime_count_per_day.index)
    crime_count_per_day.plot(color='#ff5252', label='Daily Crime Count',figsize=(20,10))
    mean_value = crime_count_per_day.mean()
    plt.axhline(y=mean_value, color='b', linestyle='--')

    x_position = crime_count_per_day.index.max()

    start_date = pd.to_datetime('2020-03-10').date()
    end_date = pd.to_datetime('2022-04-27').date()

    plt.axvline(x=pd.to_datetime('2020-03-10').date(), color='b', linestyle='--', )
    plt.axvline(x=pd.to_datetime('2022-04-27').date(), color='b', linestyle='--', )

    plt.axvspan(start_date, end_date, color='yellow', alpha=0.3)  # alpha sets the transparency

    y_min, y_max = plt.gca().get_ylim()

    # Adjust these offsets if necessary
    vertical_offset_start = y_max * 0.15  # Adjust this value to move the text up or down for 'COVID starts'
    vertical_offset_end = y_max * 0.15

    # Annotating the start and end of COVID
    plt.text(x=x_position, y=mean_value, s=f'Mean: {mean_value:.2f}', color='blue', ha='right', va='center', fontsize=12, backgroundcolor='white')
    plt.text(start_date, vertical_offset_start, 'COVID-19 Restictions Begins', ha='right', va='center', rotation=90, backgroundcolor='white')
    plt.text(end_date, vertical_offset_end, 'COVID-19 Restictions Ends', ha='left', va='center', rotation=90, backgroundcolor='white')

    plt.title('Daily Crime Count', fontsize=20,fontweight='bold')
    plt.xlabel('Date',fontsize=18,fontweight='bold')
    plt.ylabel('Crime Count',fontsize=18,fontweight='bold')

    # Marking specific dates with dots and labels
    specific_dates = [pd.to_datetime('2016-01-23').date()
                    , pd.to_datetime('2020-06-01').date()
                    , pd.to_datetime('2012-10-29').date()
                    , pd.to_datetime('2020-10-27').date()
                    ,pd.to_datetime('2010-02-10').date()]  # Replace with your actual dates
    specific_counts = crime_count_per_day.loc[specific_dates]

    plt.plot(specific_dates, specific_counts, 'o', color='black',markersize=3)

    plt.annotate(f'Blizzard - Crime Count: {specific_counts.iloc[0]}', (specific_dates[0], specific_counts.iloc[0]),
                textcoords="offset points", xytext=(0,-10), ha='center')

    plt.annotate(f'George Floyd Protest - Crime Count: {specific_counts.iloc[1]}', (specific_dates[1]
                                                                                    , specific_counts.iloc[1]),
                textcoords="offset points", xytext=(0,2), ha='center')

    plt.annotate(f'Hurricane Sandy - Crime Count: {specific_counts.iloc[2]}', (specific_dates[2]
                                                                                    , specific_counts.iloc[2]),
                textcoords="offset points", xytext=(-65,-100), ha='center',rotation=30)
    plt.annotate(f' Walter Wallace Jr Shooting - Crime Count: {specific_counts.iloc[3]}\n', (specific_dates[3]
                                                                                    , specific_counts.iloc[3]),
                textcoords="offset points", xytext=(100,-5), ha='center')
    plt.annotate(f'Winter Storm - Crime Count: {specific_counts.iloc[4]}', (specific_dates[4]
                                                                                    , specific_counts.iloc[4]),
                textcoords="offset points", xytext=(-65,-50), ha='center',rotation=15)


    is_christmas = (crime_count_per_day.index.month == 12) & (crime_count_per_day.index.day == 25)

    # Filter for Christmas dates using the mask
    christmas_dates = crime_count_per_day.index[is_christmas]
    christmas_counts = crime_count_per_day[is_christmas]

    plt.plot(christmas_dates, christmas_counts, 'o', color='green', markersize=5, label='Christmas')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    # Show the plot
    return plt

plotContainer4 = st.container()
with plotContainer4:
    st.header('Daily Crime Report Counts Analysis')
    st.subheader("Let's analyze the daily crime counts")

    plt = plot_crime_data(crime_count_per_day)
    st.pyplot(plt)
    st.markdown('**Insights**')
    st.markdown(f'*1. During COVID-19 the number of reported crimes were below the mean*')
    st.markdown(f'*2. The highest number of reported crimes were reported on 1st June 2021*')
    st.markdown(f'*3. The lowest number of reported crimes were reported on 23rd January 2016*')


mean_temp_per_day = timeseries.groupby(timeseries['Datetime'].dt.date)['temperature_2m_mean (°F)'].mean()

crime_count_per_day = timeseries.groupby(timeseries['Datetime'].dt.date).size()

# Now, let's create a DataFrame for count_temp_per_day that includes both the mean temperature and crime count
count_temp_per_day = mean_temp_per_day.to_frame(name='Mean Temperature (°F)')
count_temp_per_day['Crime Count'] = crime_count_per_day

#count_temp_per_day

plotContainer5 = st.container()
with plotContainer5:
    
    st.header('Temperature -  Crime Report Counts Analysis')
    st.subheader("Let's find out is there a correlation between temperature and crime reports")
    # Assuming count_temp_per_day is your DataFrame with 'Mean Temperature (°F)' and 'Crime Count' columns
    x = count_temp_per_day['Mean Temperature (°F)']
    y = count_temp_per_day['Crime Count']

    # Calculate the line of best fit
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='#ff5252')

    # Add the regression line
    plt.plot(x, line,'--', color='blue', label='Regression Line',linewidth=1,alpha=0.6)
    correlation = 0.384421


    # Add the correlation coefficient as text on the line
    plt.text(88, 50, f'Correlation: {correlation:.3f}', color='blue', ha='center', va='bottom')

    # Add titles and labels
    plt.title('Mean Temperature vs. Crime Count',fontsize=14, fontweight='bold')
    plt.xlabel('Mean Temperature (°F)',fontsize=14)
    plt.ylabel('Crime Count',fontsize=14)
    plt.grid(True)

    # Show the plot with the regression line
    plt.legend()
    #plt.show()
    st.pyplot(plt)
    st.markdown('**Insights**')
    st.markdown(f'*1. There is a slight positive correlation between temperature and crime reports*')

mean_rain_per_day = timeseries.groupby(timeseries['Datetime'].dt.date)['precipitation_sum (mm)'].mean()
crime_count_per_day = timeseries.groupby(timeseries['Datetime'].dt.date).size()

# Now, let's create a DataFrame for count_temp_per_day that includes both the mean temperature and crime count
count_rain_per_day = mean_rain_per_day.to_frame(name='precipitation_sum (mm)')
count_rain_per_day['Crime Count'] = crime_count_per_day

#count_rain_per_day
plotContainer6 = st.container()
with plotContainer6:
    st.header('Precipitation -  Crime Report Counts Analysis')
    st.subheader("Let's find out is there a correlation between precipitation and crime reports")
# Assuming count_temp_per_day is your DataFrame with 'Mean Temperature (°F)' and 'Crime Count' columns
    x = count_rain_per_day['precipitation_sum (mm)']
    y = count_rain_per_day['Crime Count']

    # Calculate the line of best fit
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='#ff5252')

    # Add the regression line
    plt.plot(x, line,'--', color='blue', label='Regression Line',linewidth=1,alpha=0.6)
    correlation = -0.138874


    # Add the correlation coefficient as text on the line
    plt.text(85, 50, f'Correlation: {correlation:.3f}', color='blue', ha='center', va='bottom')
    #plt.plot(x, line, color='red', label='Regression Line')

    # Add titles and labels
    plt.title('Precipitation vs. Crime Count',fontsize=14, fontweight='bold')
    plt.xlabel('Precipitation (mm)',fontsize=14)
    plt.ylabel('Crime Count',fontsize=14)
    plt.grid(True)

    # Show the plot with the regression line
    plt.legend()
    #plt.show()
    st.pyplot(plt)
    st.markdown('**Insights**')
    st.markdown(f'*1. There is a slight negative correlation between precipitation and crime reports*')


plotContainer7 = st.container()
with plotContainer7:
    st.header('Thank You!')
#map_con = st.container()

#drop_cols = ['id', 'location', 'date', 'severity', 'borough', 'age', 'class', 'mode', 'ageBand', 'vehicles']
#map_df = api_data_modified.drop(columns = drop_cols)
#print(map_df)
#map_df=data[['point_y','point_x']]
#map_df = map_df.rename(columns={'point_y': 'LATITUDE', 'point_x': 'LONGITUDE'})


#with map_con:
 # st.header('A simple map of the accident zones in london')
  #st.map(map_df, use_container_width =  True)