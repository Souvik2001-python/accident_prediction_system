import pickle
import streamlit as st
import numpy as np
from PIL import Image
import base64
import time

pipe = pickle.load(open('accident_prediction_trained_model_pipe.pkl', 'rb'))
img = Image.open('img/ac4.jpg')

# prediction function
def pred(Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level, Vehicle_driver_relation,
         Driving_experience, Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,
         Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians, Road_allignment,
         Types_of_Junction, Road_surface_type, Road_surface_conditions, Light_conditions,
         Weather_conditions, Type_of_collision, Number_of_vehicles_involved,
         Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty,
         Age_band_of_casualty, Casualty_severity, Work_of_casuality, Fitness_of_casuality,
         Pedestrian_movement, Cause_of_accident, Hour_of_Day,pipe):

    # Your prediction code here
    features = np.array([[Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level, Vehicle_driver_relation,
         Driving_experience, Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,
         Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians, Road_allignment,
         Types_of_Junction, Road_surface_type, Road_surface_conditions, Light_conditions,
         Weather_conditions, Type_of_collision, Number_of_vehicles_involved,
         Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty,
         Age_band_of_casualty, Casualty_severity, Work_of_casuality, Fitness_of_casuality,
         Pedestrian_movement, Cause_of_accident, Hour_of_Day]])
    
    try:
      results = pipe.predict(features)
      if results[0] == 2:
        return "Slight Injury"
      elif results[0] == 1:
        return "Serious Injury"
      else:
        return "Fatal Injury"
  
    except AttributeError as e:
        return "Model Version Error: Please reinstall compatible scikit-learn version."

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(
    page_title="Accident Prediction App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
    # Load CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    with open("img/ac4.jpg", "rb") as f:
      encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="centered">
        <h1>Accident Prediction App</h1>
        <p>
            India records over 1.5 lakh road accident deaths every year, with overspeeding,
            drunk driving, rule violations, and lack of helmets or seatbelts being the
            major causes. Young adults and two-wheeler riders are the most affected.
            Road safety is a shared responsibility.
        </p>
                    ✔  Follow speed limits and traffic signals ,<br>
                    ✔  Avoid distractions while driving ,<br>
                    ✔  Never drive under the influence of alcohol or drugs ,<br>
                    ✔  Always wear a seatbelt and helmet ,<br>
                    ✔  Maintain a safe distance from other vehicles ,<br>
                    ✔  Use indicators for lane changes and turns ,<br>
                    ✔  Avoid mobile use while driving ,<br>
                    ✔  Regularly maintain your vehicle ,<br>
                    ✔  Be extra cautious in adverse weather conditions ,<br>
                    ✔  Educate others about road safety
          
    </div>
""", unsafe_allow_html=True)

    # sidebar inputs
    with st.sidebar:
        st.write('## Please enter the following details to predict the severity of an accident:')
        Day_of_week = st.selectbox('Day of Week',
                                  ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
        Age_band_of_driver = st.selectbox('Age Band of Driver',
                                          ['17-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-65',
                                          '66-75', 'Over 75'])
        Sex_of_driver = st.selectbox('Sex of Driver', ['Male', 'Female', 'Unknown'])
        Educational_level = st.selectbox('Educational Level', ['Above high school', 'Junior high school', 'Elementary school',
                                                              'High school', 'Unknown', 'Illiterate', 'Writing & reading'])
        Vehicle_driver_relation = st.selectbox('Vehicle Driver Relation', ['Employee', 'Unknown', 'Owner', 'Other'])
        Driving_experience = st.selectbox('Driving Experience', ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence',
                                                                'Below 1yr', 'unknown'])
        Type_of_vehicle = st.selectbox('Type of Vehicle', ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Public (13?45 seats)',
                                                            'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q',
                                                            'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle',
                                                            'Special vehicle', 'Bicycle'])
        Owner_of_vehicle = st.selectbox('Owner of Vehicle', ['Owner', 'Governmental', 'Organization', 'Other'])
        Service_year_of_vehicle = st.selectbox('Service Year of Vehicle', ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown', 'Below 1yr'])
        Defect_of_vehicle = st.selectbox('Defect of Vehicle', ['No defect', '7', '5'])
        Area_accident_occured = st.selectbox('Area of Accident Occurred', ['Residential areas', 'Office areas', 'Recreational areas',
                                                                          'Industrial areas', 'Other', 'Church areas', 'Market areas',
                                                                          'Unknown', 'Rural village areas', 'Outside rural areas',
                                                                          'Hospital areas', 'School areas',
                                                                          'Rural village areasOffice areas', 'Recreational areas'])
        Lanes_or_Medians = st.selectbox('Lanes or Medians', ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way',
                                                            'Two-way (divided with solid lines road marking)',
                                                            'Two-way (divided with broken lines road marking)', 'Unknown'])
        Road_allignment = st.selectbox('Road Allignment', ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain',
                                                          'Escarpments', 'Tangent road with rolling terrain', 'Gentle horizontal curve',
                                                          'Tangent road with mountainous terrain and', 'Steep grade downward with mountainous terrain',
                                                          'Sharp reverse curve', 'Steep grade upward with mountainous terrain'])
        Types_of_Junction = st.selectbox('Types of Junction', ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
                                                              'X Shape'])
        Road_surface_type = st.selectbox('Road Surface Type', ['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress',
                                                              'Gravel roads', 'Other'])
        Road_surface_conditions = st.selectbox('Road Surface Conditions', ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'])
        Light_conditions = st.selectbox('Light Conditions', ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                                                            'Darkness - lights unlit'])
        Weather_conditions = st.selectbox('Weather Conditions', ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy',
                                                                'Snow', 'Unknown', 'Fog or mist'])
        Type_of_collision = st.selectbox('Type of Collision', ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
                                                              'Collision with roadside objects', 'Collision with animals', 'Other',
                                                              'Rollover', 'Fall from vehicles', 'Collision with pedestrians',
                                                              'With Train', 'Unknown'])
        Number_of_vehicles_involved = st.number_input('Number of Vehicles Involved', min_value=1, max_value=10, step=1, value=1)
        Number_of_casualties = st.number_input('Number of Casualties', min_value=1, max_value=10, step=1, value=1)
        Vehicle_movement = st.selectbox('Vehicle Movement', ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go',
                                                            'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking',
                                                            'Other', 'Entering a junction'])
        Casualty_class = st.selectbox('Casualty Class', ['na', 'Driver or rider', 'Pedestrian', 'Passenger'])
        Sex_of_casualty = st.selectbox('Sex of Casualty', ['na', 'Male', 'Female'])
        Age_band_of_casualty = st.selectbox('Age Band of Casualty', ['na', '31-50', '18-30', 'Under 18', 'Over 51', '5'])
        Casualty_severity = st.selectbox('Casualty Severity', ['na', '3', '2', '1'])
        Work_of_casuality = st.selectbox('Work of Casualty', ['Driver', 'Other', 'Unemployed', 'Employee', 'Self-employed', 'Student', 'Unknown'])
        Fitness_of_casuality = st.selectbox('Fitness of Casualty', ['Normal', 'Deaf', 'Other', 'Blind', 'NormalNormal'])
        Pedestrian_movement = st.selectbox('Pedestrian Movement', ["Not a Pedestrian",
                                                                  "Crossing from driver's nearside",
                                                                  "Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle",
                                                                  "Unknown or other",
                                                                  "Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle",
                                                                  "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)",
                                                                  "Walking along in carriageway, back to traffic",
                                                                  "Walking along in carriageway, facing traffic",
                                                                  "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle"])
        Cause_of_accident = st.selectbox('Cause of Accident', ['Moving Backward', 'Overtaking', 'Changing lane to the left',
                                                              'Changing lane to the right', 'Overloading', 'Other',
                                                              'No priority to vehicle', 'No priority to pedestrian',
                                                              'No distancing', 'Getting off the vehicle improperly',
                                                              'Improper parking', 'Overspeed', 'Driving carelessly',
                                                              'Driving at high speed', 'Driving to the left', 'Unknown',
                                                              'Overturning', 'Turnover', 'Driving under the influence of drugs',
                                                              'Drunk driving'])
        Hour_of_Day = st.selectbox('Hour of Day', [17, 1, 14, 22, 8, 15, 12, 18, 13, 20, 16, 21, 9, 10, 19, 11, 23, 7, 0, 5, 6, 4, 3, 2])


        if "last_click_time" not in st.session_state:
            st.session_state.last_click_time = 0
        current_time = time.time()
        
        # Predict button
        if st.button('Predict Accident Severity'):
          
          # Prevent spamming (2 second cooldown)
          if current_time - st.session_state.last_click_time < 2:
              st.warning("Please wait 3 seconds before clicking again...")
              st.stop()
          
          st.session_state.last_click_time = current_time

          result = pred(Day_of_week=Day_of_week,
                           Age_band_of_driver=Age_band_of_driver,
                           Sex_of_driver=Sex_of_driver,
                           Educational_level=Educational_level,
                           Vehicle_driver_relation=Vehicle_driver_relation,
                           Driving_experience=Driving_experience,
                           Type_of_vehicle=Type_of_vehicle,
                           Owner_of_vehicle=Owner_of_vehicle,
                           Service_year_of_vehicle=Service_year_of_vehicle,
                           Defect_of_vehicle=Defect_of_vehicle,
                           Area_accident_occured=Area_accident_occured,
                           Lanes_or_Medians=Lanes_or_Medians,
                           Road_allignment=Road_allignment,
                           Types_of_Junction=Types_of_Junction,
                           Road_surface_type=Road_surface_type,
                           Road_surface_conditions=Road_surface_conditions,
                           Light_conditions=Light_conditions,
                           Weather_conditions=Weather_conditions,
                           Type_of_collision=Type_of_collision,
                           Number_of_vehicles_involved=Number_of_vehicles_involved,
                           Number_of_casualties=Number_of_casualties,
                           Vehicle_movement=Vehicle_movement,
                           Casualty_class=Casualty_class,
                           Sex_of_casualty=Sex_of_casualty,
                           Age_band_of_casualty=Age_band_of_casualty,
                           Casualty_severity=Casualty_severity,
                           Work_of_casuality=Work_of_casuality,
                           Fitness_of_casuality=Fitness_of_casuality,
                           Pedestrian_movement=Pedestrian_movement,
                           Cause_of_accident=Cause_of_accident,
                           Hour_of_Day=Hour_of_Day,
                           pipe=pipe)
          
          
          # Unique ID so alert always re-renders
          alert_id = int(time.time() * 1000) 

          # st.success(f'Predicted Accident Severity: {result}')
          # st.markdown(f"""
          #     <div class="custom-alert">
          #         <strong>Predicted Accident Severity:</strong><br>
          #         {result}
          #     </div>
          # """, unsafe_allow_html=True)
          st.markdown(f"""
              <div class="custom-alert" id="alert-{alert_id}">
                  <strong>Predicted Accident Severity:</strong><br>
                  {result}
              </div>
          """, unsafe_allow_html=True)

    st.markdown(
          """
          <div class="footer">
              Developed And Maintained By Souvik Sen (Full Stack Developer @ WinQuest Online Pvt. Ltd.)
          </div>
          """,
          unsafe_allow_html=True
      )

if __name__ == "__main__":
    main()