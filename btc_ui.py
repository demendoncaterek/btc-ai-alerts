import os,requests,pandas as pd,streamlit as st,plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

ENGINE=os.getenv("ENGINE_URL")

st.set_page_config("BTC AI Dashboard",layout="wide")
st_autorefresh(interval=10000,key="r")

state=requests.get(f"{ENGINE}/state").json()
metrics=requests.get(f"{ENGINE}/metrics").json()
trades=requests.get(f"{ENGINE}/trades").json()["trades"]

st.title("🧠 BTC AI Dashboard")

c1,c2,c3,c4=st.columns(4)
c1.metric("Price",f"${state['price']:,.2f}")
c2.metric("Signal",state["signal"])
c3.metric("Confidence",f"{state['confidence']*100:.1f}%")
c4.metric("Equity",f"${metrics['equity']:,.2f}")

st.subheader("📊 Performance")
m1,m2,m3,m4=st.columns(4)
m1.metric("Win rate",f"{metrics['win_rate']*100:.1f}%")
m2.metric("Profit factor",f"{metrics['profit_factor']:.2f}")
m3.metric("Avg R",f"{metrics['avg_r']:.2f}")
m4.metric("Max DD",f"{metrics['max_dd']*100:.1f}%")

if trades:
    df=pd.DataFrame(trades)
    df["cum"]=df.pnl.cumsum()+250
    fig=go.Figure(go.Scatter(y=df.cum,mode="lines",name="Equity"))
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("🧾 Trades")
    st.dataframe(df,use_container_width=True)
