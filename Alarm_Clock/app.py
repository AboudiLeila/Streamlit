import streamlit as st
import datetime
import webbrowser
import threading
import time

alarm_time = None

def set_alarm():
    global alarm_time
    
    try:
        hour_val = int(hour)
        minute_val = int(minute)
        if time_var == "PM":
            hour_val += 12
        alarm_time = datetime.time(hour_val, minute_val)
    except ValueError:
        st.error("Please enter valid hour and minute")
        return

    threading.Thread(target=wait_and_ring_alarm).start()
    st.success(f"Alarm set for {alarm_time.strftime('%I:%M %p')}")


def wait_and_ring_alarm():
    global alarm_time
    
    while True:
        current_time = datetime.datetime.now().time()
        if current_time.hour == alarm_time.hour and current_time.minute == alarm_time.minute:
            ring_alarm()
            break
        time.sleep(1)


def ring_alarm():
    open_youtube_video()


def open_youtube_video():
    user_url = url.strip()
    default_url = "https://www.youtube.com/watch?v=eJO5HU_7_1w"
    if user_url:
        url_to_open = user_url
    else:
        url_to_open = default_url
    webbrowser.open(url_to_open)

st.title("Alarm Clock")

hour = st.number_input("Hour", min_value=1, max_value=12)
minute = st.number_input("Minute", min_value=0, max_value=59)
time_var = st.selectbox("AM/PM", ["AM", "PM"])
url = st.text_input("Insert your song's URL", placeholder="https://www.youtube.com/watch?v=eJO5HU_7_1w")
st.button("Set Alarm", on_click=set_alarm)
