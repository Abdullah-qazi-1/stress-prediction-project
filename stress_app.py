import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Stress Level Analysis & Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.2rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_cleaned_data.csv')
        return df
    except:
        try:
            df = pd.read_csv('dataset.csv')
            return df
        except:
            return None

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('stress_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# Sidebar navigation
st.sidebar.title("🧠 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Data Analysis", "🔮 Stress Prediction"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.info("**Project:** Student Stress Level Prediction using Digital Usage Patterns")

# ==================== PAGE 1: HOME ====================
if page == "🏠 Home":
    st.title("🧠 Student Stress Level Analysis & Prediction")
    st.markdown("### Understanding the Impact of Digital Usage on Student Stress")
    st.markdown("---")
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📖 Project Overview
        
        This project analyzes the relationship between **digital usage patterns** and **stress levels** 
        among university students. Using machine learning, we predict stress levels based on:
        
        - 📱 Mobile and social media usage
        - 🎮 Gaming habits
        - 📚 Study hours
        - 😴 Sleep quality
        - 🧠 Behavioral patterns
        
        **Goal:** Help students understand how their digital habits affect their mental health 
        and provide personalized recommendations for stress management.
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x300/4CAF50/FFFFFF?text=Stress+Analysis", 
                use_container_width=True)
    
    st.markdown("---")
    
    # Load data for statistics
    df = load_data()
    
    if df is not None:
        st.markdown("## 📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Students", len(df))
        
        with col2:
            st.metric("📋 Features", len(df.columns))
        
        with col3:
            avg_stress = df['recent_stress'].mean() if 'recent_stress' in df.columns else 0
            st.metric("📈 Avg Stress Level", f"{avg_stress:.2f}/5")
        
        with col4:
            avg_digital = df['total_digital_hrs'].mean() if 'total_digital_hrs' in df.columns else 0
            st.metric("⏰ Avg Digital Hours", f"{avg_digital:.1f}h")
        
        st.markdown("---")
        
        # Quick insights
        st.markdown("## 🔍 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Key Findings:
            - 📱 **High mobile usage** correlates with increased stress
            - 😴 **Poor sleep quality** is a major stress indicator
            - 🎮 **Gaming habits** show mixed effects on stress
            - 📚 **Study hours** have complex relationship with stress
            - 🧘 **Lifestyle balance** is crucial for stress management
            """)
        
        with col2:
            st.markdown("""
            ### Methodology:
            1. **Data Collection:** Survey of 33 university students
            2. **Feature Engineering:** Created composite metrics
            3. **Model Training:** Random Forest Classifier
            4. **Validation:** Cross-validation and testing
            5. **Deployment:** Interactive Streamlit application
            """)
        
        st.markdown("---")
        
        # Distribution preview
        st.markdown("## 📈 Stress Level Distribution")
        
        if 'recent_stress' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            stress_counts = df['recent_stress'].value_counts().sort_index()
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
            ax.bar(stress_counts.index, stress_counts.values, color=colors[:len(stress_counts)])
            ax.set_xlabel('Stress Level', fontsize=12)
            ax.set_ylabel('Number of Students', fontsize=12)
            ax.set_title('Distribution of Stress Levels Among Students', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.info("**Interpretation:** Most students experience moderate to high stress levels (3-4 out of 5)")
    
    else:
        st.warning("⚠️ Dataset not found. Please ensure 'final_cleaned_data.csv' is in the project directory.")

# ==================== PAGE 2: DATA ANALYSIS ====================
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis & Visualizations")
    st.markdown("### Exploring the Relationship Between Digital Usage and Stress")
    st.markdown("---")
    
    df = load_data()
    
    if df is not None:
        # Section 1: Distribution Analysis
        st.markdown("## 📈 Distribution Analysis")
        st.markdown("Understanding the basic patterns in our dataset")
        
        tab1, tab2, tab3 = st.tabs(["👥 Demographics", "📱 Digital Usage", "😴 Lifestyle"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'gender' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    gender_counts = df['gender'].value_counts()
                    ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                          colors=['#3498db', '#e74c3c'], startangle=90)
                    ax.set_title('Gender Distribution', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                    st.caption("📊 Distribution of male and female students in the study")
            
            with col2:
                if 'age' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, x='age', palette='viridis', ax=ax)
                    ax.set_title('Age Group Distribution', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Age Group', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    st.pyplot(fig)
                    plt.close()
                    st.caption("📊 Most students are in the 20-22 age group")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'mobile_hrs' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, x='mobile_hrs', palette='Reds', ax=ax)
                    ax.set_title('Daily Mobile Usage', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Hours per Day', fontsize=12)
                    ax.set_ylabel('Number of Students', fontsize=12)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                    st.caption("📱 Mobile usage varies widely among students")
            
            with col2:
                if 'social_hrs' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, x='social_hrs', palette='Blues', ax=ax)
                    ax.set_title('Daily Social Media Usage', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Hours per Day', fontsize=12)
                    ax.set_ylabel('Number of Students', fontsize=12)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                    st.caption("📲 Social media usage shows concerning patterns")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'study_hrs' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, x='study_hrs', palette='Greens', ax=ax)
                    ax.set_title('Daily Study Hours', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Hours per Day', fontsize=12)
                    ax.set_ylabel('Number of Students', fontsize=12)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                    st.caption("📚 Study hours are generally low among students")
            
            with col2:
                if 'sleep_quality' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, x='sleep_quality', palette='Purples', ax=ax, 
                                order=['Very Poor', 'Poor', 'Average', 'Good'])
                    ax.set_title('Sleep Quality', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Quality', fontsize=12)
                    ax.set_ylabel('Number of Students', fontsize=12)
                    st.pyplot(fig)
                    plt.close()
                    st.caption("😴 Sleep quality is a major concern")
        
        st.markdown("---")
        
        # Section 2: Relationship Analysis
        st.markdown("## 🔗 Stress Relationship Analysis")
        st.markdown("How different factors correlate with stress levels")
        
        if 'recent_stress' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'mobile_hrs' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, x='mobile_hrs', y='recent_stress', palette='Reds', ax=ax)
                    ax.set_title('Stress Level by Mobile Usage', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Mobile Usage (hours/day)', fontsize=12)
                    ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                    st.info("💡 **Insight:** Higher mobile usage tends to correlate with increased stress levels")
            
            with col2:
                if 'sleep_quality' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, x='sleep_quality', y='recent_stress', palette='Purples', ax=ax,
                              order=['Very Poor', 'Poor', 'Average', 'Good'])
                    ax.set_title('Stress Level by Sleep Quality', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Sleep Quality', fontsize=12)
                    ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                    st.pyplot(fig)
                    plt.close()
                    st.info("💡 **Insight:** Poor sleep quality is strongly associated with higher stress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'gender' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, x='gender', y='recent_stress', palette='Set2', ax=ax)
                    ax.set_title('Stress Level by Gender', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Gender', fontsize=12)
                    ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                    st.pyplot(fig)
                    plt.close()
                    st.info("💡 **Insight:** Gender differences in stress levels are observable")
            
            with col2:
                if 'play_games' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, x='play_games', y='recent_stress', palette='Set1', ax=ax)
                    ax.set_title('Stress Level by Gaming Habit', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Plays Games', fontsize=12)
                    ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                    st.pyplot(fig)
                    plt.close()
                    st.info("💡 **Insight:** Gaming shows mixed effects on stress levels")
        
        st.markdown("---")
        
        # Section 3: Correlation Analysis
        st.markdown("## 📉 Scatter Plot Analysis")
        st.markdown("Exploring continuous relationships with stress")
        
        if all(col in df.columns for col in ['total_digital_hrs', 'digital_addiction_score', 'lifestyle_balance', 'recent_stress']):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df['total_digital_hrs'], df['recent_stress'], 
                          color='#e74c3c', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
                ax.set_xlabel('Total Digital Hours', fontsize=12)
                ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                ax.set_title('Digital Hours vs Stress', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
                st.caption("📱 More digital time → Higher stress")
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df['digital_addiction_score'], df['recent_stress'], 
                          color='#3498db', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
                ax.set_xlabel('Digital Addiction Score', fontsize=12)
                ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                ax.set_title('Addiction Score vs Stress', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
                st.caption("🎯 Higher addiction → Higher stress")
            
            with col3:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df['lifestyle_balance'], df['recent_stress'], 
                          color='#2ecc71', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
                ax.set_xlabel('Lifestyle Balance', fontsize=12)
                ax.set_ylabel('Stress Level (1-5)', fontsize=12)
                ax.set_title('Lifestyle Balance vs Stress', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
                st.caption("⚖️ Better balance → Lower stress")
        
        st.markdown("---")
        
        # Section 4: Interactive Plotly Charts
        st.markdown("## 🎨 Interactive Visualizations")
        st.markdown("Hover over the charts for detailed information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'mobile_hrs' in df.columns and 'recent_stress' in df.columns:
                stress_by_mobile = df.groupby('mobile_hrs')['recent_stress'].mean().reset_index()
                fig = px.bar(stress_by_mobile, x='mobile_hrs', y='recent_stress',
                           title='Average Stress Level by Mobile Usage Hours',
                           color='recent_stress', color_continuous_scale='Reds',
                           labels={'recent_stress': 'Avg Stress (1-5)', 'mobile_hrs': 'Mobile Hours/Day'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'sleep_quality' in df.columns and 'recent_stress' in df.columns:
                stress_by_sleep = df.groupby('sleep_quality')['recent_stress'].mean().reset_index()
                fig = px.bar(stress_by_sleep, x='sleep_quality', y='recent_stress',
                           title='Average Stress Level by Sleep Quality',
                           color='recent_stress', color_continuous_scale='Blues',
                           labels={'recent_stress': 'Avg Stress (1-5)', 'sleep_quality': 'Sleep Quality'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ **Analysis Complete!** Use the sidebar to navigate to the Prediction page to test the model.")
    
    else:
        st.error("❌ Dataset not found. Please ensure 'final_cleaned_data.csv' is in the project directory.")

# ==================== PAGE 3: STRESS PREDICTION ====================
elif page == "🔮 Stress Prediction":
    st.title("🔮 Stress Level Prediction")
    st.markdown("### Predict your stress level based on digital usage and lifestyle patterns")
    st.markdown("---")
    
    model = load_model()
    
    if model is not None:
        st.success("✅ Model loaded successfully!")
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📱 Digital Usage")
            mobile_num = st.slider("Mobile Usage (hours/day)", 0.0, 12.0, 3.0, 0.5,
                                   help="Average daily mobile phone usage in hours")
            social_num = st.slider("Social Media (hours/day)", 0.0, 12.0, 2.0, 0.5,
                                  help="Average daily social media usage in hours")
            gaming_num = st.slider("Gaming (hours/day)", 0.0, 12.0, 1.0, 0.5,
                                  help="Average daily gaming hours")
            
            play_games_num = st.selectbox("Do you play games?", 
                                         options=["No", "Yes"],
                                         help="Do you play games on mobile or PC?")
            play_games_num = 1 if play_games_num == "Yes" else 0
        
        with col2:
            st.subheader("📚 Study & Sleep")
            study_num = st.slider("Study Hours (hours/day)", 0.0, 12.0, 4.0, 0.5,
                                 help="Daily study hours excluding class time")
            sleep_num = st.slider("Sleep Quality (1-4)", 1, 4, 3,
                                 help="1=Very Poor, 2=Poor, 3=Average, 4=Good")
            
            st.subheader("👤 Personal Info")
            gender_num = st.selectbox("Gender", options=["Male", "Female"])
            gender_num = 0 if gender_num == "Male" else 1
            
            age_num = st.selectbox("Age Group", options=["17-19", "20-22"])
            age_num = 0 if age_num == "17-19" else 1
        
        with col3:
            st.subheader("🧠 Behavioral Patterns")
            
            check_after_wake_num = st.selectbox("Check phone after waking?",
                                               options=["No", "Yes"])
            check_after_wake_num = 1.0 if check_after_wake_num == "Yes" else 0.0
            
            urge_without_notif = st.select_slider("Urge to check phone without notification",
                                                 options=["Never", "Rarely", "Sometimes", "Often", "Always"],
                                                 value="Sometimes")
            urge_map = {"Never": 0, "Rarely": 0.25, "Sometimes": 0.5, "Often": 0.75, "Always": 1}
            urge_without_notif_num = urge_map[urge_without_notif]
            
            mobile_increases_stress_num = st.selectbox("Does mobile increase your stress?",
                                                       options=["No", "Sometimes", "Yes"])
            stress_map = {"No": 0, "Sometimes": 0.5, "Yes": 1}
            mobile_increases_stress_num = stress_map[mobile_increases_stress_num]
            
            reduced_attention_num = st.selectbox("Has digital usage reduced attention span?",
                                                options=["No", "Yes"])
            reduced_attention_num = 1 if reduced_attention_num == "Yes" else 0
            
            academic_pressure = st.selectbox("Do you feel academic pressure?",
                                            options=["No", "Yes"])
            academic_pressure = 1 if academic_pressure == "Yes" else 0
            
            current_stress = st.slider("Current Stress Level (1-5)", 1, 5, 3,
                                      help="Rate your current stress: 1=Low, 5=High")
        
        st.markdown("---")
        
        # Calculate composite features
        total_digital_hrs = mobile_num + social_num + gaming_num
        digital_addiction_score = total_digital_hrs + check_after_wake_num + urge_without_notif_num
        lifestyle_balance = sleep_num + study_num - total_digital_hrs
        
        # Show calculated metrics
        with st.expander("📊 View Calculated Metrics"):
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Digital Hours", f"{total_digital_hrs:.1f}")
            with metric_col2:
                st.metric("Digital Addiction Score", f"{digital_addiction_score:.2f}")
            with metric_col3:
                st.metric("Lifestyle Balance", f"{lifestyle_balance:.1f}")
        
        # Predict button
        if st.button("🔮 Predict Stress Level"):
            # Prepare input in exact order as model expects
            input_data = np.array([[
                mobile_num,
                social_num,
                gaming_num,
                study_num,
                sleep_num,
                total_digital_hrs,
                digital_addiction_score,
                lifestyle_balance,
                gender_num,
                age_num,
                play_games_num,
                check_after_wake_num,
                urge_without_notif_num,
                mobile_increases_stress_num,
                reduced_attention_num,
                academic_pressure,
                current_stress
            ]])
            
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.markdown("---")
                st.subheader("📋 Prediction Result")
                
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                
                with result_col2:
                    # Map prediction to stress level
                    stress_levels = {
                        1: ("😊 Very Low Stress", "success", "You're managing stress very well! Keep up the good habits."),
                        2: ("🙂 Low Stress", "success", "Good stress management. Continue your healthy lifestyle!"),
                        3: ("😐 Moderate Stress", "warning", "Consider implementing stress management techniques."),
                        4: ("😟 High Stress", "warning", "Time to focus on stress reduction strategies."),
                        5: ("😰 Very High Stress", "error", "⚠️ Consider seeking professional help if needed.")
                    }
                    
                    level_text, level_type, message = stress_levels.get(prediction, stress_levels[3])
                    
                    if level_type == "success":
                        st.success(f"### {level_text}")
                        st.balloons()
                    elif level_type == "warning":
                        st.warning(f"### {level_text}")
                    else:
                        st.error(f"### {level_text}")
                    
                    st.info(message)
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Personalized Recommendations")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("**Digital Wellness Tips:**")
                    if total_digital_hrs > 8:
                        st.markdown("- 📵 **Reduce screen time** (currently high at {:.1f}h)".format(total_digital_hrs))
                    if check_after_wake_num == 1:
                        st.markdown("- ⏰ **Avoid checking phone** immediately after waking")
                    if urge_without_notif_num > 0.5:
                        st.markdown("- 🧘 **Practice mindfulness** to reduce phone dependency")
                    st.markdown("- 🌙 Use **blue light filters** before bed")
                    st.markdown("- 📱 Set **app time limits** on your phone")
                
                with rec_col2:
                    st.markdown("**Lifestyle Balance:**")
                    if sleep_num < 3:
                        st.markdown("- 😴 **Improve sleep quality** (currently low)")
                    if study_num < 3:
                        st.markdown("- 📚 **Increase study hours** for better academic performance")
                    if lifestyle_balance < 0:
                        st.markdown("- ⚖️ **Balance digital time** with productive activities")
                    st.markdown("- 🏃 **Regular physical exercise** (30 min/day)")
                    st.markdown("- 🥗 Maintain a **healthy diet**")
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check all inputs and try again.")
    
    else:
        st.warning("⚠️ Could not load the model. Please ensure 'stress_model.pkl' exists.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🧠 Stress Level Predictor | Made with Streamlit</p>
        <p style='font-size: 0.8rem;'>⚠️ Educational purposes only. Consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)