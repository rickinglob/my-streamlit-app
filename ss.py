import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import smtplib
import time
import pandas_ta as ta
from email.mime.text import MIMEText
from datetime import datetime
from ta.trend import MACD, ADXIndicator, CCIIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Dict, Tuple
from ta.momentum import (RSIIndicator, StochasticOscillator, ROCIndicator, TSIIndicator)

# Set page config as the first Streamlit command
st.set_page_config(page_title="Momentum Scanner Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (updated to style sidebar alerts)
st.markdown("""
    <style>
    .reportview-container {
        background: #1a1a1a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
    }
    .stSlider > div > div > div {
        background-color: #333;
        border-color: #555;
    }
    .stExpander > div > div > div {
        background-color: #2a2a2a;
        border-color: #555;
    }
    .stSelectbox > div > div > div {
        background-color: #2a2a2a;
        border-color: #555;
    }
    .stSidebar .sidebar-content {
        background-color: #2a2a2a;
        border-right: 1px solid #555;
    }
    .main-content {
        padding: 20px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stWarning {
        background-color: #856404;
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
    }
    .sidebar .stWarning {
        margin: 5px 0;
    }
    /* New styles for indicator pills */
    .indicator-pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
        font-size: 12px;
        font-weight: bold;
    }
   .indicator-strong-buy {
       background-color: #2E7D32;
       color: white;
   }

   .indicator-buy {
       background-color: #4CAF50;
       color: white;
   }

   .indicator-weak-buy {
       background-color: #81C784;
       color: white;
   }

   .indicator-weak-sell {
       background-color: #EF9A9A;
       color: white;
   }

   .indicator-sell {
       background-color: #f44336;
       color: white;
   }

   .indicator-strong-sell {
       background-color: #C62828;
       color: white;
    }
    .indicator-neutral {
        background-color: #9e9e9e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

INDEX_CONSTITUENTS = {

    "Nifty Futures":["^NSEI", "^NSEBANK", "^BSESN", "^NSMIDCP", "NIFTY_FIN_SERVICE.NS", "AARTIIND.NS", "ABB.NS", "ABCAPITAL.NS",
    "ABFRL.NS", "ACC.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", "ANGELONE.NS", "APLAPOLLO.NS",
    "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATGL.NS", "AUBANK.NS", "AUROPHARMA.NS", 
    "AXISBANK.NS",  "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BEL.NS","BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "BSE.NS", "BSOFT.NS", "CAMS.NS", "CANBK.NS", "CDSL.NS", "CESC.NS",
    "CGPOWER.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS",
    "CROMPTON.NS", "CUMMINSIND.NS", "CYIENT.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", "DIVISLAB.NS",
    "DIXON.NS", "DLF.NS", "DMART.NS", "HINDALCO.NS", "DRREDDY.NS","ICICIBANK.NS", "EICHERMOT.NS", "INDIGO.NS", "INDUSINDBK.NS",
    "KALYANKJIL.NS", "ESCORTS.NS", "KPITTECH.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "PERSISTENT.NS", "SBICARD.NS",
    "SHREECEM.NS", "SRF.NS", "SYNGENE.NS", "TATASTEEL.NS", "TORNTPHARM.NS", "UPL.NS", "GLENMARK.NS", "GMRAIRPORT.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HFCL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HUDCO.NS", "IIFL.NS", "INDUSTOWER.NS",
    "INFY.NS", "IOC.NS", "IRB.NS", "IRCTC.NS", "IREDA.NS", "IRFC.NS", "JIOFIN.NS", "JKCEMENT.NS", "JUBLFOOD.NS", "LICHSGFIN.NS",
    "LODHA.NS", "LT.NS", "LTIM.NS", "LTTS.NS", "M&M.NS", "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS", "MCX.NS", "MOTHERSON.NS",
    "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "NBCC.NS", "NCC.NS", "NHPC.NS", "OBEROIRLTY.NS", "OIL.NS", "ONGC.NS",
    "PATANJALI.NS", "PEL.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS", "POLICYBZR.NS", "POWERGRID.NS", "RAMCOCEM.NS", "ICICIPRULI.NS",
    "RECLTD.NS", "HDFCBANK.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SJVN.NS", "INDIANB.NS", 
    "SUNPHARMA.NS", "SUPREME.NS", "TATACHEM.NS", "TATACOMM.NS", "JSL.NS", "TATACONSUM.NS", "JSWENERGY.NS", "TATAELXSI.NS", "TATAMOTORS.NS", 
    "TATATECH.NS", "KEI.NS", "TCS.NS", "TECHM.NS", "TIINDIA.NS", "LUPIN.NS", "TITAGARH.NS", "TITAN.NS", "MANAPPURAM.NS", "TVSMOTOR.NS", "ULTRACEMCO.NS", "UNITDSPR.NS", "VBL.NS",
    "WIPRO.NS", "ICICIGI.NS", "YESBANK.NS", "NMDC.NS", "NTPC.NS", "PAYTM.NS", "PETRONET.NS", "IDEA.NS", "INDHOTEL.NS", 
    "POLYCAB.NS", "SBILIFE.NS", "LICI.NS", "LTF.NS", "SIEMENS.NS", "SOLARINDS.NS", "UNIONBANK.NS", "VOLTAS.NS", "ITC.NS", "JINDALSTEL.NS",
    "JSWSTEEL.NS", "VEDL.NS", "M&MFIN.NS", "MGL.NS", "KOTAKBANK.NS", "MFSL.NS", "PRESTIGE.NS", "SAIL.NS", "TATAPOWER.NS", 
    "RELIANCE.NS", "TRENT.NS", "SONACOMS.NS", "ZYDUSLIFE.NS", "LAURUSLABS.NS", "NATIONALUM.NS", "NESTLEIND.NS", "NYKAA.NS", "OFSS.NS",
    "PAGEIND.NS", "PIIND.NS", "POONAWALLA.NS", "RBLBANK.NS", "ZOMATO.NS", "PHOENIXLTD.NS", "CASTROLIND.NS", "GLAND.NS",
    "TORNTPOWER.NS", "AMARAJA.NS"],
    
    "Nifty 50": ["ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BEL.NS","BPCL.NS","BHARTIARTL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS",
    "HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","ITC.NS","INDUSINDBK.NS",
    "INFY.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS","NTPC.NS","NESTLEIND.NS","ONGC.NS","POWERGRID.NS",
    "RELIANCE.NS","SBILIFE.NS","SHRIRAMFIN.NS","SBIN.NS","SUNPHARMA.NS","TCS.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TECHM.NS","TITAN.NS","TRENT.NS","ULTRACEMCO.NS","WIPRO.NS"],

    "Nifty 500":["360ONE.NS", "3MINDIA.NS", "ABB.NS", "ACC.NS", "AIAENG.NS", "APLAPOLLO.NS", "AUBANK.NS", "AADHARHFC.NS",
    "AARTIIND.NS", "AAVAS.NS", "ABBOTINDIA.NS", "ACE.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS",
    "ADANIPORTS.NS", "ADANIPOWER.NS", "ATGL.NS", "AWL.NS", "ABCAPITAL.NS", "ABFRL.NS", "ABREL.NS", "ABSLAMC.NS",
    "AEGISLOG.NS", "AFFLE.NS", "AJANTPHARM.NS", "AKUMS.NS", "APLLTD.NS", "ALKEM.NS", "ALKYLAMINE.NS", "ALOKINDS.NS",
    "ARE&M.NS", "AMBER.NS", "AMBUJACEM.NS", "ANANDRATHI.NS", "ANANTRAJ.NS", "ANGELONE.NS", "APARINDS.NS",
    "APOLLOHOSP.NS", "APOLLOTYRE.NS", "APTUS.NS", "ACI.NS", "ASAHIINDIA.NS", "ASHOKLEY.NS", "ASIANPAINT.NS",
    "ASTERDM.NS", "ASTRAZEN.NS", "ASTRAL.NS", "ATUL.NS", "AUROPHARMA.NS", "AVANTIFEED.NS", "DMART.NS", "AXISBANK.NS",
    "BASF.NS", "BEML.NS", "BLS.NS", "BSE.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
    "BALAMINES.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "MAHABANK.NS",
    "BATAINDIA.NS", "BAYERCROP.NS", "BERGEPAINT.NS", "BDL.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS", "BPCL.NS",
    "BHARTIARTL.NS", "BHARTIHEXA.NS", "BIKAJI.NS", "BIOCON.NS", "BIRLACORPN.NS", "BSOFT.NS", "BLUEDART.NS",
    "BLUESTARCO.NS", "BBTC.NS", "BOSCHLTD.NS", "BRIGADE.NS", "BRITANNIA.NS", "MAPMYINDIA.NS", "CCL.NS", "CESC.NS",
    "CGPOWER.NS", "CIEINDIA.NS", "CRISIL.NS", "CAMPUS.NS", "CANFINHOME.NS", "CANBK.NS", "CAPLIPOINT.NS", "CGCL.NS",
    "CARBORUNIV.NS", "CASTROLIND.NS", "CEATLTD.NS", "CELLO.NS", "CENTRALBK.NS", "CDSL.NS", "CENTURYPLY.NS", "CERA.NS",
    "CHALET.NS", "CHAMBLFERT.NS", "CHEMPLASTS.NS", "CHENNPETRO.NS", "CHOLAHLDNG.NS", "CHOLAFIN.NS", "CIPLA.NS",
    "CUB.NS", "CLEAN.NS", "COALINDIA.NS", "COCHINSHIP.NS", "COFORGE.NS", "COLPAL.NS", "CAMS.NS", "CONCORDBIO.NS",
    "CONCOR.NS", "COROMANDEL.NS", "CRAFTSMAN.NS", "CREDITACC.NS", "CROMPTON.NS", "CUMMINSIND.NS", "CYIENT.NS",
    "DLF.NS", "DOMS.NS", "DABUR.NS", "DALBHARAT.NS", "DATAPATTNS.NS", "DEEPAKFERT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS",
    "DEVYANI.NS", "DIVISLAB.NS", "DIXON.NS", "LALPATHLAB.NS", "DRREDDY.NS", "EIDPARRY.NS", "EIHOTEL.NS", "EASEMYTRIP.NS",
    "EICHERMOT.NS", "ELECON.NS", "ELGIEQUIP.NS", "EMAMILTD.NS", "EMCURE.NS", "ENDURANCE.NS", "ENGINERSIN.NS",
    "EQUITASBNK.NS", "ERIS.NS", "ESCORTS.NS", "EXIDEIND.NS", "NYKAA.NS", "FEDERALBNK.NS", "FACT.NS", "FINEORG.NS",
    "FINCABLES.NS", "FINPIPE.NS", "FSL.NS", "FIVESTAR.NS", "FORTIS.NS", "GRINFRA.NS", "GAIL.NS", "GVT&D.NS",
    "GMRAIRPORT.NS", "GRSE.NS", "GICRE.NS", "GILLETTE.NS", "GLAND.NS", "GLAXO.NS", "GLENMARK.NS", "MEDANTA.NS",
    "GODIGIT.NS", "GPIL.NS", "GODFRYPHLP.NS", "GODREJAGRO.NS", "GODREJCP.NS", "GODREJIND.NS", "GODREJPROP.NS",
    "GRANULES.NS", "GRAPHITE.NS", "GRASIM.NS", "GESHIP.NS", "GRINDWELL.NS", "GAEL.NS", "FLUOROCHEM.NS", "GUJGASLTD.NS",
    "GMDCLTD.NS", "GNFC.NS", "GPPL.NS", "GSFC.NS", "GSPL.NS", "HEG.NS", "HBLENGINE.NS", "HCLTECH.NS", "HDFCAMC.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HFCL.NS", "HAPPSTMNDS.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HSCL.NS", "HINDALCO.NS",
    "HAL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "POWERINDIA.NS", "HOMEFIRST.NS",
    "HONASA.NS", "HONAUT.NS", "HUDCO.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "ISEC.NS", "IDBI.NS",
    "IDFCFIRSTB.NS", "IFCI.NS", "IIFL.NS", "INOXINDIA.NS", "IRB.NS", "IRCON.NS", "ITC.NS", "ITI.NS", "INDGN.NS",
    "INDIACEM.NS", "INDIAMART.NS", "INDIANB.NS", "IEX.NS", "INDHOTEL.NS", "IOC.NS", "IOB.NS", "IRCTC.NS", "IRFC.NS",
    "IREDA.NS", "IGL.NS", "INDUSTOWER.NS", "INDUSINDBK.NS", "NAUKRI.NS", "INFY.NS", "INOXWIND.NS", "INTELLECT.NS",
    "INDIGO.NS", "IPCALAB.NS", "JBCHEPHARM.NS", "JKCEMENT.NS", "JBMA.NS", "JKLAKSHMI.NS", "JKTYRE.NS", "JMFINANCIL.NS",
    "JSWENERGY.NS", "JSWINFRA.NS", "JSWSTEEL.NS", "JPPOWER.NS", "J&KBANK.NS", "JINDALSAW.NS", "JSL.NS", "JINDALSTEL.NS",
    "JIOFIN.NS", "JUBLFOOD.NS", "JUBLINGREA.NS", "JUBLPHARMA.NS", "JWL.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS", "JYOTICNC.NS",
    "KPRMILL.NS", "KEI.NS", "KNRCON.NS", "KPITTECH.NS", "KSB.NS", "KAJARIACER.NS", "KPIL.NS", "KALYANKJIL.NS",
    "KANSAINER.NS", "KARURVYSYA.NS", "KAYNES.NS", "KEC.NS", "KFINTECH.NS", "KIRLOSBROS.NS", "KIRLOSENG.NS", "KOTAKBANK.NS",
    "KIMS.NS", "LTF.NS", "LTTS.NS", "LICHSGFIN.NS", "LTIM.NS", "LT.NS", "LATENTVIEW.NS", "LAURUSLABS.NS", "LEMONTREE.NS",
    "LICI.NS", "LINDEINDIA.NS", "LLOYDSME.NS", "LUPIN.NS", "MMTC.NS", "MRF.NS", "LODHA.NS", "MGL.NS", "MAHSEAMLES.NS",
    "M&MFIN.NS", "M&M.NS", "MAHLIFE.NS", "MANAPPURAM.NS", "MRPL.NS", "MANKIND.NS", "MARICO.NS", "MARUTI.NS",
    "MASTEK.NS", "MFSL.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "METROBRAND.NS", "METROPOLIS.NS", "MINDACORP.NS",
    "MSUMI.NS", "MOTILALOFS.NS", "MPHASIS.NS", "MCX.NS", "MUTHOOTFIN.NS", "NATCOPHARM.NS", "NBCC.NS", "NCC.NS",
    "NHPC.NS", "NLCINDIA.NS", "NMDC.NS", "NSLNISP.NS", "NTPC.NS", "NH.NS", "NATIONALUM.NS", "NAVINFLUOR.NS",
    "NESTLEIND.NS", "NETWEB.NS", "NETWORK18.NS", "NEWGEN.NS", "NAM-INDIA.NS", "NUVAMA.NS", "NUVOCO.NS", "OBEROIRLTY.NS",
    "ONGC.NS", "OIL.NS", "OLECTRA.NS", "PAYTM.NS", "OFSS.NS", "POLICYBZR.NS", "PCBL.NS", "PIIND.NS", "PNBHOUSING.NS",
    "PNCINFRA.NS", "PTCIL.NS", "PVRINOX.NS", "PAGEIND.NS", "PATANJALI.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS",
    "PHOENIXLTD.NS", "PIDILITIND.NS", "PEL.NS", "PPLPHARMA.NS", "POLYMED.NS", "POLYCAB.NS", "POONAWALLA.NS",
    "PFC.NS", "POWERGRID.NS", "PRAJIND.NS", "PRESTIGE.NS", "PGHH.NS", "PNB.NS", "QUESS.NS", "RRKABEL.NS", "RBLBANK.NS",
    "RECLTD.NS", "RHIM.NS", "RITES.NS", "RADICO.NS", "RVNL.NS", "RAILTEL.NS", "RAINBOW.NS", "RAJESHEXPO.NS",
    "RKFORGE.NS", "RCF.NS", "RATNAMANI.NS", "RTNINDIA.NS", "RAYMOND.NS", "REDINGTON.NS", "RELIANCE.NS", "ROUTE.NS",
    "SBFC.NS", "SBICARD.NS", "SBILIFE.NS", "SJVN.NS", "SKFINDIA.NS", "SRF.NS", "SAMMAANCAP.NS", "MOTHERSON.NS",
    "SANOFI.NS", "SAPPHIRE.NS", "SAREGAMA.NS", "SCHAEFFLER.NS", "SCHNEIDER.NS", "SCI.NS", "SHREECEM.NS", "RENUKA.NS",
    "SHRIRAMFIN.NS", "SHYAMMETL.NS", "SIEMENS.NS", "SIGNATURE.NS", "SOBHA.NS", "SOLARINDS.NS", "SONACOMS.NS",
    "SONATSOFTW.NS", "STARHEALTH.NS", "SBIN.NS", "SAIL.NS", "SWSOLAR.NS", "SUMICHEM.NS", "SPARC.NS", "SUNPHARMA.NS",
    "SUNTV.NS", "SUNDARMFIN.NS", "SUNDRMFAST.NS", "SUPREMEIND.NS", "SUVENPHAR.NS", "SUZLON.NS", "SWANENERGY.NS",
    "SYNGENE.NS", "SYRMA.NS", "TBOTEK.NS", "TVSMOTOR.NS", "TVSSCS.NS", "TANLA.NS", "TATACHEM.NS", "TATACOMM.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAINVEST.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS",
    "TATATECH.NS", "TTML.NS", "TECHM.NS", "TECHNOE.NS", "TEJASNET.NS", "NIACL.NS", "RAMCOCEM.NS", "THERMAX.NS",
    "TIMKEN.NS", "TITAGARH.NS", "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TRIDENT.NS", "TRIVENI.NS",
    "TRITURBINE.NS", "TIINDIA.NS", "UCOBANK.NS", "UNOMINDA.NS", "UPL.NS", "UTIAMC.NS", "UJJIVANSFB.NS", "ULTRACEMCO.NS",
    "UNIONBANK.NS", "UBL.NS", "UNITDSPR.NS", "USHAMART.NS", "VGUARD.NS", "VIPIND.NS", "DBREALTY.NS", "VTL.NS",
    "VARROC.NS", "VBL.NS", "MANYAVAR.NS", "VEDL.NS", "VIJAYA.NS", "VINATIORGA.NS", "IDEA.NS", "VOLTAS.NS",
    "WELCORP.NS", "WELSPUNLIV.NS", "WESTLIFE.NS", "WHIRLPOOL.NS", "WIPRO.NS", "YESBANK.NS", "ZFCVINDIA.NS",
    "ZEEL.NS", "ZENSARTECH.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "ECLERX.NS"]
}

def get_constituents(index_name: str) -> list:
    """
    Retrieves the constituents of a given index.
    """
    return INDEX_CONSTITUENTS.get(index_name, [])

# Expanded cache duration
@st.cache_data(ttl=900)
def get_asset_data(ticker, period="3mo", interval="1H"):
    try:      
        asset = yf.Ticker(ticker)
        data = asset.history(period=period, interval=interval)
        if data.empty:
            st.sidebar.warning(f"No data found for {ticker}. Check the ticker symbol.")
            return None
        return data
    except Exception as e:
        st.sidebar.error(f"Error fetching data for {ticker}: {e}")
        return None

# Cache for Streamlit (if used)
try:
    from streamlit import cache_data
except ImportError:
    cache_data = lambda x: x  # Fallback if not using Streamlit

def display_signal_indicator(signal):
    """Creates stylized signal indicators based on the signal type"""
    
    signal_classes = {
        "STRONG BUY": "indicator-pill indicator-strong-buy",
        "BUY": "indicator-pill indicator-buy",
        "WEAK BUY": "indicator-pill indicator-weak-buy",
        "NEUTRAL": "indicator-pill indicator-neutral",
        "WEAK SELL": "indicator-pill indicator-weak-sell",
        "SELL": "indicator-pill indicator-sell",
        "STRONG SELL": "indicator-pill indicator-strong-sell"
    }
    
    if signal in signal_classes:
        return f'<span class="{signal_classes[signal]}">{signal}</span>'
    else:
        return signal

@cache_data(ttl=900)
def calculate_advanced_indicators(data: pd.DataFrame, short_window: Optional[int] = None,
                                long_window: Optional[int] = None, rsi_window: int = 14,
                                macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """
    Calculate a comprehensive set of technical indicators with enhancements.

    Parameters:
    - data: DataFrame with OHLCV (Open, High, Low, Close, Volume) data
    - short_window, long_window: Optional dynamic SMA windows (default adaptive)
    - rsi_window, macd_fast, macd_slow, macd_signal: Indicator parameters

    Returns:
    - DataFrame with calculated indicators
    """


    # Validate input
    if data is None or data.empty or len(data) < 5 or not all(col in data for col in ["Close", "High", "Low"]):
        raise ValueError("Invalid input data: Must contain OHLC data with at least 5 rows")


    # Make a copy and handle missing data
    df = data.copy()

    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

    volatility = atr / df["Close"]

    if short_window is None:
        short_window = max(5, min(20, int(10 / volatility.mean())))
    if long_window is None:
        long_window = short_window * 4
    
    # Traditional SMA-based momentum
    if short_window is None:
        # Make sure volatility has valid values
        valid_volatility = volatility.dropna()
        if len(valid_volatility) > 0:
            short_window = max(5, min(20, int(10 / valid_volatility.mean())))
        else:
            # Default if volatility calculation fails
            short_window = 5
    
    if long_window is None:
        long_window = short_window * 4
    
    # Adjust windows if insufficient data
    data_points = len(df)
    if data_points < long_window:
        # Scale down windows proportionally
        scale_factor = data_points / (long_window + 1)  # +1 to ensure we have at least one valid point
        short_window = max(2, int(short_window * scale_factor))
        long_window = max(5, int(long_window * scale_factor))
    
    # Traditional SMA-based momentum
    df["Short_SMA"] = df["Close"].rolling(window=short_window, min_periods=1).mean()
    df["Long_SMA"] = df["Close"].rolling(window=long_window, min_periods=1).mean()
    
    # Calculate momentum with fallback logic
    df["Momentum_SMA"] = df["Short_SMA"] - df["Long_SMA"]
    
    # Handle edge cases
    if df["Momentum_SMA"].isna().all():
        # If all momentum values are NaN, set to a small default value
        df["Momentum_SMA"] = 0
    else:
        # Fill remaining NaNs with the closest valid value
        df["Momentum_SMA"] = df["Momentum_SMA"].fillna(method='bfill')
        # If bfill doesn't work for the last values, use ffill as backup
        df["Momentum_SMA"] = df["Momentum_SMA"].fillna(method='ffill')
        # As a last resort, fill with 0
        df["Momentum_SMA"] = df["Momentum_SMA"].fillna(0)
        

    # RSI - Relative Strength Index
    df["RSI"] = RSIIndicator(df["Close"], window=rsi_window).rsi()

    # MACD - Moving Average Convergence Divergence
    macd = MACD(df["Close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()

    # Stochastic Oscillator
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Rate of Change (ROC)
    df["ROC"] = ROCIndicator(df["Close"], window=12).roc()

    # True Strength Index (TSI)
    df["TSI"] = TSIIndicator(df["Close"], window_slow=25, window_fast=13).tsi()

    # KAMA - Kaufman Adaptive Moving Average
    df["KAMA"] = ta.kama(df["Close"], length=10, fast=2, slow=30) # Use length for periods parameter

    # Bollinger Bands
    bollinger = BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bollinger.bollinger_hband()
    df["BB_Lower"] = bollinger.bollinger_lband()
    df["BB_Middle"] = bollinger.bollinger_mavg()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Percent"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # Average True Range - volatility indicator
    df["ATR"] = atr
    df["ATR_Percent"] = df["ATR"] / df["Close"] * 100

    # ADX - Average Directional Index - trend strength
    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["ADX"] = adx.adx()
    df["DI_Plus"] = adx.adx_pos()
    df["DI_Minus"] = adx.adx_neg()

    # CCI - Commodity Channel Index
    df["CCI"] = CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci()

    # Volume-based indicators
    if "Volume" in df.columns and not df["Volume"].isnull().all():
        df["MFI"] = MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index()
        df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
        df["OBV_ROC"] = df["OBV"].pct_change(periods=10) * 100
        df["VWAP"] = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    else:
        print("Skipping volume-based indicators due to missing Volume column or all NaN values.")

    # Enhanced Supertrend with smoothing
    atr_smooth = df["ATR"].ewm(span=7).mean()  # Smoothed ATR

    atr_factor = 3
    df["basic_upper"] = (df["High"] + df["Low"]) / 2 + atr_factor * atr_smooth
    df["basic_lower"] = (df["High"] + df["Low"]) / 2 - atr_factor * atr_smooth

    # Initialize supertrend columns
    df["supertrend_upper"] = np.nan  
    df["supertrend_lower"] = np.nan  
    df["supertrend"] = 0  
    
    # Calculate initial supertrend direction
    if len(df) > 0:
        # Make sure basic_upper is not NaN in the first row
        if not pd.isna(df["basic_upper"].iloc[0]):
            df.loc[df.index[0], "supertrend"] = 1 if df["Close"].iloc[0] <= df["basic_upper"].iloc[0] else -1
            # Initialize the first values for supertrend_upper and supertrend_lower
            if df["supertrend"].iloc[0] == 1:
                df.loc[df.index[0], "supertrend_upper"] = df["basic_upper"].iloc[0]
            else:
                df.loc[df.index[0], "supertrend_lower"] = df["basic_lower"].iloc[0]
    
    # Iterative Supertrend calculation with NaN handling
        for i in range(1, len(df)):
            # Skip calculation if basic bands are NaN
            if pd.isna(df["basic_upper"].iloc[i]) or pd.isna(df["basic_lower"].iloc[i]):
                df.loc[df.index[i], "supertrend"] = df["supertrend"].iloc[i-1]
                continue
                
            prev_supertrend = df["supertrend"].iloc[i - 1]
            curr_close = df["Close"].iloc[i]
            curr_basic_upper = df["basic_upper"].iloc[i]
            curr_basic_lower = df["basic_lower"].iloc[i]
            
            # Get previous values with NaN handling
            prev_upper = df["supertrend_upper"].iloc[i - 1]
            prev_lower = df["supertrend_lower"].iloc[i - 1]
            prev_close = df["Close"].iloc[i - 1]
            
            # Determine supertrend upper/lower based on previous trend
            if prev_supertrend == 1:
                # Only compare with previous if it's not NaN
                if not pd.isna(prev_upper):
                    df.loc[df.index[i], "supertrend_upper"] = curr_basic_upper if (
                        curr_basic_upper < prev_upper or prev_close > prev_upper
                    ) else prev_upper
                else:
                    df.loc[df.index[i], "supertrend_upper"] = curr_basic_upper
                df.loc[df.index[i], "supertrend_lower"] = np.nan
            else:
                # Only compare with previous if it's not NaN
                if not pd.isna(prev_lower):
                    df.loc[df.index[i], "supertrend_lower"] = curr_basic_lower if (
                        curr_basic_lower > prev_lower or prev_close < prev_lower
                    ) else prev_lower
                else:
                    df.loc[df.index[i], "supertrend_lower"] = curr_basic_lower
                df.loc[df.index[i], "supertrend_upper"] = np.nan
            
            # Update supertrend direction with NaN handling
            if prev_supertrend == 1 and curr_close <= df["supertrend_upper"].iloc[i]:
                df.loc[df.index[i], "supertrend"] = 1
            elif prev_supertrend == 1 and curr_close > df["supertrend_upper"].iloc[i]:
                df.loc[df.index[i], "supertrend"] = -1
            elif prev_supertrend == -1 and curr_close >= df["supertrend_lower"].iloc[i]:
                df.loc[df.index[i], "supertrend"] = -1
            elif prev_supertrend == -1 and curr_close < df["supertrend_lower"].iloc[i]:
                df.loc[df.index[i], "supertrend"] = 1
            else:
                df.loc[df.index[i], "supertrend"] = prev_supertrend


    # Exponential Moving Averages
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # Hull Moving Average (HMA)
    wma_half = df["Close"].rolling(window=5).apply(lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    wma_full = df["Close"].rolling(window=10).apply(lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    df["HMA"] = pd.Series(2 * wma_half - wma_full).rolling(window=3).apply(lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True)

    # New Indicators
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(df["High"], df["Low"], window1=9, window2=26, window3=52)
    df["Ichimoku_A"] = ichimoku.ichimoku_a()
    df["Ichimoku_B"] = ichimoku.ichimoku_b()
    df["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
    df["Ichimoku_Conv"] = ichimoku.ichimoku_conversion_line()

    # Calculate Choppiness Index
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
    tr = tr.fillna(0)  # Handle NaN from shift
    atr_sum = tr.rolling(window=14).sum()
    epsilon = 1e-9
    high_low_range = df["High"].rolling(window=14).max() - df["Low"].rolling(window=14).min() + epsilon  # Prevent div by zero

    ratio = atr_sum / high_low_range
    df["Choppiness"] = np.where(ratio > 0, 100 * np.log10(ratio) / np.log10(14), np.nan)

    # Momentum Metrics
    df["EMA_Cross"] = (df["EMA_9"] > df["EMA_21"]).astype(int)
    df["Golden_Cross"] = (df["EMA_50"] > df["EMA_200"]).astype(int)
    df["RSI_Stoch_Combined"] = (df["RSI"] + df["Stoch_K"]) / 2
    df["Trend_Strength"] = (df["ADX"] / 100) * (df["supertrend"] * 50 + 50)

    print(f"Output data type: {type(df)}")
    print(f"Output data shape: {df.shape}")
    print(f"Output data head:\n{df.head()}")
    return df 
    print(df)

def train_ml_momentum_model(data: pd.DataFrame, lookback: int = 100) -> Tuple[float, RandomForestClassifier]:
    """Train a Random Forest model to predict momentum."""
    features = data[["RSI", "MACD", "Stoch_K", "ADX", "supertrend", "CCI", "ATR_Percent", "BB_Percent", "EMA_Cross"]]
    target = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    train_data = features.iloc[-lookback:].dropna()
    train_target = target.iloc[-lookback:].dropna()
    
    if len(train_data) < 10:
        return 50.0, None
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_data, train_target)
    
    latest = features.iloc[-1:].ffill()
    pred_proba = model.predict_proba(latest)[0, 1] * 100  # Probability of price increase
    return pred_proba, model

@cache_data(ttl=900)
def calculate_momentum_score(data: pd.DataFrame, higher_tf_data: Optional[pd.DataFrame] = None, 
                            weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict]:
    """
    Calculate momentum score for a given stock data with optimized performance.
    Integrates ML predictions and multi-timeframe analysis for enhanced accuracy.
    
    Args:
        data (pd.DataFrame): OHLCV dataframe with indicators
        higher_tf_data (pd.DataFrame, optional): Higher timeframe data for confirmation
        weights (Dict[str, float], optional): Custom indicator weights
        
    Returns:
        Tuple[float, Dict]: Momentum score (0-100) and signal details
    """
    # Basic validation with early returns
    if data is None or data.empty:
        return 0, {"Error": {"value": "No data", "signal": "neutral"}}
    
    if len(data) < 26:
        return 50, {"Warning": {"value": "Insufficient data", "signal": "neutral"}}

    # Default weights with proper initialization
    if weights is None:
        weights = {
            "sma_momentum": 8, "macd": 12, "rsi": 12, "stochastic": 8, "supertrend": 15,
            "adx": 10, "cci": 5, "mfi": 8, "bb_position": 5, "ema_cross": 5, "ichimoku": 7, "vwap": 5
        }

    # Get latest data point efficiently
    latest = data.iloc[-1].copy()
    signals = {}
    score_components = {}

    # Helper function to get indicator value safely
    def get_safe_value(column, default=0):
        if column in latest and not pd.isna(latest[column]):
            return latest[column]
        return default

    # Calculate individual component scores - each with error handling
    try:
        # SMA Momentum
        sma_momentum = get_safe_value("Momentum_SMA")
        # Scale factor calculation with safeguards
        close_price = max(0.01, get_safe_value("Close", 1))  # Avoid div by zero
        scale_factor = 1000 if close_price < 100 else 500 if close_price < 1000 else 250
        sma_score = min(100, max(0, 50 + (sma_momentum / close_price * scale_factor)))
        score_components["sma_momentum"] = sma_score
        signals["SMA Momentum"] = {"value": f"{sma_momentum:.2f}", "signal": "buy" if sma_momentum > 0 else "sell"}
    except Exception as e:
        score_components["sma_momentum"] = 50
        signals["SMA Momentum"] = {"value": "Error", "signal": "neutral"}

    try:
        # MACD
        macd_hist = get_safe_value("MACD_Histogram")
        atr = max(0.01, get_safe_value("ATR", 1))  # Avoid div by zero
        macd_score = min(100, max(0, 50 + (macd_hist / atr * 50)))
        score_components["macd"] = macd_score
        signals["MACD"] = {"value": f"{macd_hist:.2f}", "signal": "buy" if macd_hist > 0 else "sell"}
    except Exception as e:
        score_components["macd"] = 50
        signals["MACD"] = {"value": "Error", "signal": "neutral"}

    try:
        # RSI
        rsi_val = get_safe_value("RSI", 50)
        # Better scaling for RSI that focuses on extremes
        if rsi_val <= 30:
            rsi_score = 100 - ((30 - rsi_val) * 1.5)  # Strong buy below 30
        elif rsi_val >= 70:
            rsi_score = 100 - ((rsi_val - 70) * 1.5)  # Strong sell above 70
        else:
            # Neutral zone scaling - closer to 50 is more neutral
            rsi_score = 100 - (min(abs(rsi_val - 30), abs(rsi_val - 70)) / 20 * 100)
        
        rsi_score = min(100, max(0, rsi_score))
        score_components["rsi"] = rsi_score
        
        if rsi_val < 30:
            rsi_signal = "buy"
        elif rsi_val > 70:
            rsi_signal = "sell"
        else:
            # More nuanced neutral signal
            rsi_signal = "neutral" if 45 <= rsi_val <= 55 else "weak buy" if rsi_val < 45 else "weak sell"
            
        signals["RSI"] = {"value": f"{rsi_val:.2f}", "signal": rsi_signal}
    except Exception as e:
        score_components["rsi"] = 50
        signals["RSI"] = {"value": "Error", "signal": "neutral"}

    try:
        # Stochastic
        stoch_k = get_safe_value("Stoch_K", 50)
        stoch_d = get_safe_value("Stoch_D", 50)
        
        # More sophisticated stochastic scoring
        if stoch_k < 20:
            stoch_score = 80 + (20 - stoch_k) * 1  # Oversold
        elif stoch_k > 80:
            stoch_score = 20 - (stoch_k - 80) * 1  # Overbought
        else:
            stoch_score = 50 - (stoch_k - 50) * 0.8  # Neutral zone
        
        # Add crossover influence
        if stoch_k > stoch_d and stoch_k < 80:
            stoch_score += 10  # Bullish crossover
        elif stoch_k < stoch_d and stoch_k > 20:
            stoch_score -= 10  # Bearish crossover
            
        stoch_score = min(100, max(0, stoch_score))
        score_components["stochastic"] = stoch_score
        
        if stoch_k < 20 and stoch_k > stoch_d:
            stoch_signal = "buy"  # Oversold with bullish crossover
        elif stoch_k > 80 and stoch_k < stoch_d:
            stoch_signal = "sell"  # Overbought with bearish crossover
        elif stoch_k < 20:
            stoch_signal = "weak buy"  # Just oversold
        elif stoch_k > 80:
            stoch_signal = "weak sell"  # Just overbought
        else:
            stoch_signal = "neutral"
            
        signals["Stochastic"] = {"value": f"K:{stoch_k:.2f} D:{stoch_d:.2f}", "signal": stoch_signal}
    except Exception as e:
        score_components["stochastic"] = 50
        signals["Stochastic"] = {"value": "Error", "signal": "neutral"}

    try:
        # Supertrend - critically important indicator
        supertrend_val = get_safe_value("supertrend", 0)
        supertrend_score = 100 if supertrend_val == 1 else 0
        score_components["supertrend"] = supertrend_score
        signals["Supertrend"] = {"value": "Bullish" if supertrend_val == 1 else "Bearish", 
                                "signal": "buy" if supertrend_val == 1 else "sell"}
    except Exception as e:
        score_components["supertrend"] = 50
        signals["Supertrend"] = {"value": "Error", "signal": "neutral"}

    try:
        # ADX - trend strength
        adx_val = get_safe_value("ADX", 0)
        di_plus = get_safe_value("DI_Plus", 0)
        di_minus = get_safe_value("DI_Minus", 0)
        
        # More sophisticated ADX scaling
        if adx_val < 15:
            # Weak trend - less confidence
            adx_score = adx_val * 2  
        elif adx_val < 25:
            # Developing trend
            adx_score = 30 + (adx_val - 15) * 2
        else:
            # Strong trend - high confidence
            adx_score = 50 + min(50, (adx_val - 25) * 1.5)
        
        # Direction component
        if di_plus > di_minus:
            adx_direction = 1  # Bullish
        elif di_minus > di_plus:
            adx_direction = -1  # Bearish
        else:
            adx_direction = 0  # Neutral
            
        # Combine strength and direction
        if adx_direction == 1:
            adx_score = min(100, adx_score)
        elif adx_direction == -1:
            adx_score = min(100, 100 - adx_score)
        else:
            adx_score = 50
            
        score_components["adx"] = adx_score
        
        # Determine signal based on both strength and direction
        if adx_val > 25 and di_plus > di_minus:
            adx_signal = "buy"
        elif adx_val > 25 and di_plus < di_minus:
            adx_signal = "sell"
        elif adx_val > 15:
            adx_signal = "weak buy" if di_plus > di_minus else "weak sell" if di_plus < di_minus else "neutral"
        else:
            adx_signal = "neutral"  # Too weak to make a decision
            
        signals["ADX"] = {"value": f"{adx_val:.2f}", "signal": adx_signal}
    except Exception as e:
        score_components["adx"] = 50
        signals["ADX"] = {"value": "Error", "signal": "neutral"}

    try:
        # CCI
        cci_val = get_safe_value("CCI", 0)
        
        # Better CCI scaling with extreme recognition
        if cci_val > 200:
            cci_score = max(0, 100 - (cci_val - 200) / 10)  # Extremely overbought
        elif cci_val > 100:
            cci_score = 25  # Overbought
        elif cci_val < -200:
            cci_score = min(100, 100 + (cci_val + 200) / 10)  # Extremely oversold
        elif cci_val < -100:
            cci_score = 75  # Oversold
        else:
            cci_score = 50 + (cci_val / 200 * 50)  # Normal range
            
        cci_score = min(100, max(0, cci_score))
        score_components["cci"] = cci_score
        
        if cci_val > 100:
            cci_signal = "sell"
        elif cci_val < -100:
            cci_signal = "buy"
        else:
            cci_signal = "neutral"
            
        signals["CCI"] = {"value": f"{cci_val:.2f}", "signal": cci_signal}
    except Exception as e:
        score_components["cci"] = 50
        signals["CCI"] = {"value": "Error", "signal": "neutral"}

    try:
        # MFI - Money Flow Index
        mfi_val = get_safe_value("MFI", 50)
        
        # MFI scoring similar to RSI but with focus on extremes
        if mfi_val <= 20:
            mfi_score = 90 + (20 - mfi_val) / 2  # Strong buy below 20
        elif mfi_val >= 80:
            mfi_score = 10 - (mfi_val - 80) / 2  # Strong sell above 80
        else:
            # Neutral scaling
            mfi_score = 50 - (mfi_val - 50)
            
        mfi_score = min(100, max(0, mfi_score))
        score_components["mfi"] = mfi_score
        
        if mfi_val < 20:
            mfi_signal = "buy"
        elif mfi_val > 80:
            mfi_signal = "sell"
        else:
            mfi_signal = "neutral"
            
        signals["MFI"] = {"value": f"{mfi_val:.2f}", "signal": mfi_signal}
    except Exception as e:
        score_components["mfi"] = 50
        signals["MFI"] = {"value": "Error", "signal": "neutral"}

    try:
        # Bollinger Band Position
        bb_percent = get_safe_value("BB_Percent", 0.5)
        
        # More nuanced BB scoring
        if bb_percent <= 0.05:
            bb_score = 95  # Extremely oversold
        elif bb_percent <= 0.2:
            bb_score = 75  # Oversold
        elif bb_percent >= 0.95:
            bb_score = 5   # Extremely overbought
        elif bb_percent >= 0.8:
            bb_score = 25  # Overbought
        else:
            # Linear scaling in the middle
            bb_score = 50 - (bb_percent - 0.5) * 100
            
        bb_score = min(100, max(0, bb_score))
        score_components["bb_position"] = bb_score
        
        if bb_percent < 0.2:
            bb_signal = "buy"
        elif bb_percent > 0.8:
            bb_signal = "sell"
        else:
            bb_signal = "neutral"
            
        signals["BB Position"] = {"value": f"{bb_percent:.2f}", "signal": bb_signal}
    except Exception as e:
        score_components["bb_position"] = 50
        signals["BB Position"] = {"value": "Error", "signal": "neutral"}

    try:
        # EMA Cross
        ema_cross = get_safe_value("EMA_Cross", 0)
        golden_cross = get_safe_value("Golden_Cross", 0)
        
        # Enhanced EMA cross score with golden cross factor
        ema_score = 100 if ema_cross == 1 and golden_cross == 1 else 75 if ema_cross == 1 else 25 if golden_cross == 1 else 0
        score_components["ema_cross"] = ema_score
        
        signals["EMA Cross"] = {"value": "Bullish" if ema_cross == 1 else "Bearish", 
                               "signal": "buy" if ema_cross == 1 else "sell"}
    except Exception as e:
        score_components["ema_cross"] = 50
        signals["EMA Cross"] = {"value": "Error", "signal": "neutral"}

    try:
        # Ichimoku Cloud
        ichimoku_score = 50  # Default neutral
        
        if all(col in latest for col in ["Ichimoku_A", "Ichimoku_B", "Ichimoku_Conv", "Ichimoku_Base"]):
            price_above_cloud = latest["Close"] > max(latest["Ichimoku_A"], latest["Ichimoku_B"])
            cloud_bullish = latest["Ichimoku_A"] > latest["Ichimoku_B"]
            tenkan_above_kijun = latest["Ichimoku_Conv"] > latest["Ichimoku_Base"]
            
            # More nuanced Ichimoku scoring
            if price_above_cloud and tenkan_above_kijun and cloud_bullish:
                ichimoku_score = 100  # Strong bullish
            elif price_above_cloud and (tenkan_above_kijun or cloud_bullish):
                ichimoku_score = 75   # Bullish
            elif not price_above_cloud and not tenkan_above_kijun and not cloud_bullish:
                ichimoku_score = 0    # Strong bearish
            elif not price_above_cloud and (not tenkan_above_kijun or not cloud_bullish):
                ichimoku_score = 25   # Bearish
            
            score_components["ichimoku"] = ichimoku_score
            
            # Ichimoku signal determination
            if ichimoku_score >= 75:
                ichimoku_signal = "buy"
            elif ichimoku_score <= 25:
                ichimoku_signal = "sell"
            else:
                ichimoku_signal = "neutral"
                
            signals["Ichimoku"] = {"value": f"Cloud: {'Bullish' if cloud_bullish else 'Bearish'}", 
                                  "signal": ichimoku_signal}
        else:
            score_components["ichimoku"] = 50
            signals["Ichimoku"] = {"value": "Not Available", "signal": "neutral"}
    except Exception as e:
        score_components["ichimoku"] = 50
        signals["Ichimoku"] = {"value": "Error", "signal": "neutral"}

    try:
        # VWAP
        vwap_score = 50  # Default neutral
        
        if "VWAP" in latest:
            vwap_diff_percent = (latest["Close"] - latest["VWAP"]) / latest["VWAP"] * 100
            
            # More scaled VWAP scoring
            if vwap_diff_percent > 3:
                vwap_score = 100  # Strong bullish
            elif vwap_diff_percent > 0:
                vwap_score = 75   # Bullish
            elif vwap_diff_percent < -3:
                vwap_score = 0    # Strong bearish
            elif vwap_diff_percent < 0:
                vwap_score = 25   # Bearish
            
            score_components["vwap"] = vwap_score
            
            # VWAP signal
            if vwap_score >= 75:
                vwap_signal = "buy"
            elif vwap_score <= 25:
                vwap_signal = "sell"
            else:
                vwap_signal = "neutral"
                
            signals["VWAP"] = {"value": f"{latest['VWAP']:.2f}", "signal": vwap_signal}
        else:
            score_components["vwap"] = 50
            signals["VWAP"] = {"value": "Not Available", "signal": "neutral"}
    except Exception as e:
        score_components["vwap"] = 50
        signals["VWAP"] = {"value": "Error", "signal": "neutral"}

    # Calculate Market Condition Adjustments
    try:
        # Choppiness Index (reduces confidence in choppy markets)
        choppiness = get_safe_value("Choppiness", 50)
        # Scale: 0-100 where >60 is choppy, <40 is trending
        choppiness_factor = max(0.5, 1 - (max(0, choppiness - 50) / 50))
    except Exception as e:
        choppiness_factor = 1.0  # Neutral if not available
        
    try:
        # Volatility adjustment using ATR%
        atr_percent = get_safe_value("ATR_Percent", 1)
        # Higher ATR% means higher volatility - adjust confidence
        volatility_adjustment = max(0.5, min(1.5, 1 / (max(0.1, atr_percent) / 2)))
    except Exception as e:
        volatility_adjustment = 1.0  # Neutral if not available

    # Calculate base weighted score
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    base_score = sum(score_components.get(k, 50) * normalized_weights.get(k, 0) for k in weights.keys())
    
    # Apply market condition adjustments
    adjusted_score = base_score * choppiness_factor * volatility_adjustment
    
    # Machine Learning integration
    try:
        ml_score = get_safe_value("ML_Prediction", 50)
        signals["ML Prediction"] = {"value": f"{ml_score:.2f}", 
                                   "signal": "buy" if ml_score > 60 else "sell" if ml_score < 40 else "neutral"}
        final_score = adjusted_score * 0.7 + ml_score * 0.3  # Blend: 70% technical, 30% ML
    except Exception as e:
        final_score = adjusted_score  # Fallback to technical only
        
    # Multi-timeframe confirmation
    if higher_tf_data is not None and isinstance(higher_tf_data, pd.DataFrame) and not higher_tf_data.empty and len(higher_tf_data) >= 26:
        try:
            # Recursively calculate higher timeframe score without ML and multi-TF to avoid infinite recursion
            higher_tf_weights = {k: weights.get(k, 0) * 1.5 if k in ["supertrend", "adx", "ichimoku"] else weights.get(k, 0) 
                              for k in weights.keys()}
            higher_tf_score, _ = calculate_momentum_score(data=higher_tf_data, weights=higher_tf_weights)
            
            # Alignment boost/penalty
            if (final_score > 50 and higher_tf_score > 60) or (final_score < 50 and higher_tf_score < 40):
                # Strong alignment - boost
                final_score = min(100, final_score * 1.2)
                signals["Timeframe Alignment"] = {"value": "Strong Alignment", "signal": "confirmed"}
            elif (final_score > 50 and higher_tf_score < 40) or (final_score < 50 and higher_tf_score > 60):
                # Strong disagreement - penalty
                final_score = max(0, final_score * 0.8)
                signals["Timeframe Alignment"] = {"value": "Conflicting Signals", "signal": "warning"}
            else:
                signals["Timeframe Alignment"] = {"value": "Neutral", "signal": "neutral"}
        except Exception as e:
            signals["Timeframe Alignment"] = {"value": "Error", "signal": "neutral"}
            
    # Ensure score is within bounds
    final_score = min(100, max(0, final_score))
    
    if final_score >= 75:
        momentum_signal = "STRONG BUY"
    elif final_score >= 65:
        momentum_signal = "BUY"
    elif final_score >= 55:
        momentum_signal = "WEAK BUY"
    elif final_score > 45:  # Changed to > to avoid overlap with WEAK SELL
        momentum_signal = "NEUTRAL"
    elif final_score > 35:  # Changed to > to avoid overlap with SELL
        momentum_signal = "WEAK SELL"
    elif final_score > 25:  # Changed to > to avoid overlap with STRONG SELL
        momentum_signal = "SELL"
    else:
        momentum_signal = "STRONG SELL"
        
    signals["Overall Momentum"] = {"value": f"{final_score:.2f}", "signal": momentum_signal}
    
    return final_score, signals

def send_alert_email(subject, body, to_email, from_email, password):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
        st.sidebar.success(f"Email alert sent to {to_email}!")
    except Exception as e:
        st.sidebar.error(f"Failed to send email alert: {str(e)}")

# Add custom indicator weights adjustment section
def add_weight_adjustment_section():
    with st.expander("Customize Indicator Weights"):
        col1, col2 = st.columns(2)
        weights = {}
        with col1:
            weights["sma_momentum"] = st.slider("SMA Momentum", 0, 20, 10, 1)
            weights["macd"] = st.slider("MACD", 0, 20, 15, 1)
            weights["rsi"] = st.slider("RSI", 0, 20, 15, 1)
            weights["stochastic"] = st.slider("Stochastic", 0, 20, 10, 1)
            weights["supertrend"] = st.slider("Supertrend", 0, 20, 15, 1)
        with col2:
            weights["adx"] = st.slider("ADX", 0, 20, 10, 1)
            weights["cci"] = st.slider("CCI", 0, 20, 5, 1)
            weights["mfi"] = st.slider("MFI", 0, 20, 10, 1)
            weights["bb_position"] = st.slider("Bollinger Position", 0, 20, 5, 1)
            weights["ema_cross"] = st.slider("EMA Cross", 0, 20, 5, 1)
        
        return weights
            
def check_alerts(ticker, data, momentum_score, signals, 
                momentum_threshold, rsi_threshold_high, rsi_threshold_low, 
                enable_email_alerts, to_email, from_email, password):
    """Check for various alerts and display in sidebar."""
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    
    if data is None or data.empty:
        return
    
    latest_price = data["Close"].iloc[-1]
    latest_data = data.iloc[-1]
    
    with st.sidebar:
        # 1. Momentum Score Alert
        if abs(momentum_score - 50) > momentum_threshold:
            direction = "bullish" if momentum_score > 50 else "bearish"
            alert_msg = f"🚨 Momentum Alert for {ticker}: {direction.upper()} momentum ({momentum_score:.2f}) exceeds threshold!"
            if alert_msg not in st.session_state.alerts:
                st.session_state.alerts.append(alert_msg)
                st.warning(alert_msg)
                if enable_email_alerts and to_email and from_email and password:
                    send_alert_email(
                        f"Momentum Alert for {ticker}",
                        f"Strong {direction} momentum ({momentum_score:.2f}) detected for {ticker}. Price: ₹{latest_price:.2f}",
                        to_email, from_email, password
                    )

        # 2. RSI Alert
        rsi = latest_data["RSI"] if "RSI" in latest_data and not pd.isna(latest_data["RSI"]) else 50
        if rsi > rsi_threshold_high or rsi < rsi_threshold_low:
            condition = "overbought" if rsi > rsi_threshold_high else "oversold"
            alert_msg = f"🚨 RSI Alert for {ticker}: RSI {rsi:.2f} is {condition}!"
            if alert_msg not in st.session_state.alerts:
                st.session_state.alerts.append(alert_msg)
                st.warning(alert_msg)
                if enable_email_alerts and to_email and from_email and password:
                    send_alert_email(
                        f"RSI Alert for {ticker}",
                        f"RSI: {rsi:.2f} is {condition} for {ticker}. Price: ₹{latest_price:.2f}",
                        to_email, from_email, password
                    )
        
        # 3. Supertrend Signal Change
        if len(data) >= 2:
            current_supertrend = latest_data["supertrend"] if "supertrend" in latest_data and not pd.isna(latest_data["supertrend"]) else 0
            prev_supertrend = data["supertrend"].iloc[-2] if "supertrend" in data.columns and not pd.isna(data["supertrend"].iloc[-2]) else 0
            
            if current_supertrend != prev_supertrend:
                signal = "BUY" if current_supertrend == 1 else "SELL"
                alert_msg = f"🚨 Supertrend Signal Change for {ticker}: {signal} signal triggered!"
                if alert_msg not in st.session_state.alerts:
                    st.session_state.alerts.append(alert_msg)
                    st.warning(alert_msg)
                    if enable_email_alerts and to_email and from_email and password:
                        send_alert_email(
                            f"Supertrend Alert for {ticker}",
                            f"Supertrend {signal} signal for {ticker}. Price: ₹{latest_price:.2f}",
                            to_email, from_email, password
                        )
        
        # 4. MACD Crossover
        if len(data) >= 2:
            current_macd = latest_data["MACD"] if "MACD" in latest_data and not pd.isna(latest_data["MACD"]) else 0
            current_signal = latest_data["MACD_Signal"] if "MACD_Signal" in latest_data and not pd.isna(latest_data["MACD_Signal"]) else 0
            prev_macd = data["MACD"].iloc[-2] if "MACD" in data.columns and not pd.isna(data["MACD"].iloc[-2]) else 0
            prev_signal = data["MACD_Signal"].iloc[-2] if "MACD_Signal" in data.columns and not pd.isna(data["MACD_Signal"].iloc[-2]) else 0
            
            current_above = current_macd > current_signal
            prev_above = prev_macd > prev_signal
            
            if current_above != prev_above:
                signal = "BUY" if current_above else "SELL"
                alert_msg = f"🚨 MACD Crossover for {ticker}: {signal} signal!"
                if alert_msg not in st.session_state.alerts:
                    st.session_state.alerts.append(alert_msg)
                    st.warning(alert_msg)
                    if enable_email_alerts and to_email and from_email and password:
                        send_alert_email(
                            f"MACD Crossover for {ticker}",
                            f"MACD {signal} signal for {ticker}. Price: ₹{latest_price:.2f}",
                            to_email, from_email, password)


def plot_advanced_stock_data(ticker, data, max_points=500):
    """
    Plot advanced stock data with optimized performance.
    """
    if data is None or data.empty:
        st.warning(f"Could not display graph for {ticker}.")
        return

    # Calculate indicators only once
    with st.spinner("Calculating technical indicators..."):
        try:
            data_for_graph = calculate_advanced_indicators(data)
            if data_for_graph is None or data_for_graph.empty:
                st.warning(f"Insufficient data for {ticker} to calculate indicators.")
                return

            # Handle large datasets with windowing
            if len(data_for_graph) > max_points:
                st.info(f"Limiting display to most recent {max_points} data points for performance.")
                display_data = data_for_graph.iloc[-max_points:]
            else:
                display_data = data_for_graph

            # Calculate momentum scores efficiently
            momentum_scores = []
            signals_list = []
            min_data_points = 20

            if len(data_for_graph) > min_data_points:
                with st.spinner("Calculating momentum scores..."):
                    for i in range(min_data_points, len(data_for_graph)):
                        if i % 10 == 0:  # Update every 10 points for efficiency
                            subset = data_for_graph.iloc[:i + 1]
                            score, signals = calculate_momentum_score(subset)
                            momentum_scores.append(score)
                            signals_list.append(signals.get("Overall Momentum", {}).get("signal", "neutral"))
                        else:
                            if momentum_scores:
                                momentum_scores.append(momentum_scores[-1])
                                signals_list.append(signals_list[-1])
                            else:
                                score, signals = calculate_momentum_score(data_for_graph.iloc[:i + 1])
                                momentum_scores.append(score)
                                signals_list.append(signals.get("Overall Momentum", {}).get("signal", "neutral"))

            padding = [np.nan] * (len(data_for_graph) - len(momentum_scores))
            data_for_graph["Momentum_Score"] = padding + momentum_scores
            display_data = data_for_graph.iloc[-len(display_data):]
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return

    # Create price chart with optimized traces
    fig = go.Figure()  # Initialize figure here

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=display_data.index,
        open=display_data['Open'],
        high=display_data['High'],
        low=display_data['Low'],
        close=display_data['Close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    # Add EMA lines
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['EMA_9'],
        mode='lines',
        name='EMA 9',
        line=dict(color='#f06292', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['EMA_21'],
        mode='lines',
        name='EMA 21',
        line=dict(color='#29b6f6', width=1)
    ))

    # Add Bollinger Bands
    if 'BB_Upper' in display_data.columns and not display_data['BB_Upper'].isnull().all():
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)'
        ))

    # Add Supertrend
    if 'supertrend_upper' in display_data.columns and not display_data['supertrend_upper'].isnull().all():
        st_upper_df = display_data[['supertrend_upper']].dropna()
        st_lower_df = display_data[['supertrend_lower']].dropna()

        if not st_upper_df.empty:
            fig.add_trace(go.Scatter(
                x=st_upper_df.index,
                y=st_upper_df['supertrend_upper'],
                mode='lines',
                name='Supertrend Upper',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1)
            ))

        if not st_lower_df.empty:
            fig.add_trace(go.Scatter(
                x=st_lower_df.index,
                y=st_lower_df['supertrend_lower'],
                mode='lines',
                name='Supertrend Lower',
                line=dict(color='rgba(0, 255, 0, 0.5)', width=1)
            ))

    # Update layout
    fig.update_layout(
        title=f'{ticker} - Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode='x unified'
    )

    # Add volume
    if 'Volume' in display_data.columns and not display_data['Volume'].isnull().all():
        colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in display_data.iterrows()]
        fig.add_trace(go.Bar(
            x=display_data.index,
            y=display_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5,
            yaxis='y2'
        ))
        fig.update_layout(
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False)
        )

    st.plotly_chart(fig, use_container_width=True)

    # Tabs for additional indicators
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Momentum Score"])
    with tab1:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#7b1fa2', width=2)
        ))
        rsi_fig.add_trace(go.Scatter(
            x=[display_data.index[0], display_data.index[-1]],
            y=[70, 70],
            mode='lines',
            name='Overbought',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash')
        ))
        rsi_fig.add_trace(go.Scatter(
            x=[display_data.index[0], display_data.index[-1]],
            y=[30, 30],
            mode='lines',
            name='Oversold',
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash')
        ))
        rsi_fig.update_layout(
            title='RSI (Relative Strength Index)',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            height=300,
            template='plotly_dark',
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        st.plotly_chart(rsi_fig, use_container_width=True)

    with tab2:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#64b5f6', width=2)
        ))
        macd_fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff7043', width=1)
        ))
        macd_colors = ['rgba(0, 255, 0, 0.5)' if val >= 0 else 'rgba(255, 0, 0, 0.5)' 
                      for val in display_data['MACD_Histogram']]
        macd_fig.add_trace(go.Bar(
            x=display_data.index,
            y=display_data['MACD_Histogram'],
            name='Histogram',
            marker_color=macd_colors
        ))
        macd_fig.update_layout(
            title='MACD (Moving Average Convergence Divergence)',
            xaxis_title='Date',
            yaxis_title='MACD Value',
            height=300,
            template='plotly_dark',
            showlegend=False
        )
        st.plotly_chart(macd_fig, use_container_width=True)

    with tab3:
        momentum_fig = go.Figure()
        momentum_data = display_data[['Momentum_Score']].dropna()
        if not momentum_data.empty:
            momentum_fig.add_trace(go.Scatter(
                x=momentum_data.index,
                y=momentum_data['Momentum_Score'],
                mode='lines',
                name='Momentum Score',
                line=dict(color='#ffb300', width=2)
            ))
            momentum_fig.add_shape(
                type="rect", x0=momentum_data.index[0], y0=60, x1=momentum_data.index[-1], y1=100,
                fillcolor="rgba(0, 255, 0, 0.1)", line=dict(width=0), layer="below"
            )
            momentum_fig.add_shape(
                type="rect", x0=momentum_data.index[0], y0=0, x1=momentum_data.index[-1], y1=40,
                fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0), layer="below"
            )
            momentum_fig.add_trace(go.Scatter(
                x=[momentum_data.index[0], momentum_data.index[-1]],
                y=[50, 50],
                mode='lines',
                name='Neutral',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash')
            ))
            momentum_fig.update_layout(
                title='Technical Momentum Score',
                xaxis_title='Date',
                yaxis_title='Score (0-100)',
                height=300,
                template='plotly_dark',
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )
            st.plotly_chart(momentum_fig, use_container_width=True)
    """
    Plot advanced stock data with optimized performance.

    Args:
        ticker (str): Stock ticker symbol
        data (pd.DataFrame): OHLCV dataframe
        max_points (int): Maximum number of points to display for performance
    """
    if data is None or data.empty:
        st.warning(f"Could not display graph for {ticker}.")
        return

    # Calculate indicators only once
    with st.spinner("Calculating technical indicators..."):
        try:
            # Use cached function to avoid recalculation
            data_for_graph = calculate_advanced_indicators(data)
            if data_for_graph is None or data_for_graph.empty:
                st.warning(f"Insufficient data for {ticker} to calculate indicators.")
                return

            # Handle large datasets with windowing
            if len(data_for_graph) > max_points:
                st.info(f"Limiting display to most recent {max_points} data points for performance.")
                display_data = data_for_graph.iloc[-max_points:]
            else:
                display_data = data_for_graph

            # Calculate momentum scores efficiently - vectorized approach when possible
            momentum_scores = []
            signals_list = []

            # Calculate minimum data points needed for valid indicators
            min_data_points = 20

            # Pre-calculate full dataset momentum scores
            if len(data_for_graph) > min_data_points:
                # Use vectorized operations where possible
                with st.spinner("Calculating momentum scores..."):
                    for i in range(min_data_points, len(data_for_graph)):
                        if i % 10 == 0:  # Update only every 10 points for efficiency
                            subset = data_for_graph.iloc[:i + 1]
                            score, signals = calculate_momentum_score(subset)
                            momentum_scores.append(score)
                            signals_list.append(signals.get("Overall Momentum", {}).get("signal", "neutral"))
                        else:
                            # For non-milestone points, use the last calculated value for efficiency
                            if momentum_scores:
                                momentum_scores.append(momentum_scores[-1])
                                signals_list.append(signals_list[-1])
                            else:
                                score, signals = calculate_momentum_score(data_for_graph.iloc[:i + 1])
                                momentum_scores.append(score)
                                signals_list.append(signals.get("Overall Momentum", {}).get("signal", "neutral"))

            # Add momentum scores to the dataframe
            padding = [np.nan] * (len(data_for_graph) - len(momentum_scores))
            data_for_graph["Momentum_Score"] = padding + momentum_scores

            # Ensure display data has momentum scores
            display_data = data_for_graph.iloc[-len(display_data):]
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return

    # Create price chart with optimized traces
    fig = go.Figure() # Create the figure *before* the conditional statement

    # Add candlestick chart - core visualization
    fig.add_trace(go.Candlestick(
        x=display_data.index,
        open=display_data['Open'],
        high=display_data['High'],
        low=display_data['Low'],
        close=display_data['Close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    # Add selective EMA traces - most important indicators
    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['EMA_9'],
        mode='lines',
        name='EMA 9',
        line=dict(color='#f06292', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=display_data.index,
        y=display_data['EMA_21'],
        mode='lines',
        name='EMA 21',
        line=dict(color='#29b6f6', width=1)
    ))

    # Add Bollinger Bands with optimized rendering
    if 'BB_Upper' in display_data.columns and not display_data['BB_Upper'].isnull().all():
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)'
        ))

    # Add Supertrend conditionally - avoid rendering if not available
    if 'supertrend_upper' in display_data.columns and not display_data['supertrend_upper'].isnull().all():
        # Filter out NaN values for better performance
        st_upper_df = display_data[['supertrend_upper']].dropna()
        st_lower_df = display_data[['supertrend_lower']].dropna()

        if not st_upper_df.empty:
            fig.add_trace(go.Scatter(
                x=st_upper_df.index,
                y=st_upper_df['supertrend_upper'],
                mode='lines',
                name='Supertrend Upper',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1)
            ))

        if not st_lower_df.empty:
            fig.add_trace(go.Scatter(
                x=st_lower_df.index,
                y=st_lower_df['supertrend_lower'],
                mode='lines',
                name='Supertrend Lower',
                line=dict(color='rgba(0, 255, 0, 0.5)', width=1)
            ))

    # Optimize layout configuration
    fig.update_layout(
        title=f'{ticker} - Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=50, b=10),  # Tighter margins
        hovermode='x unified'  # Better hover experience
    )

    # Add volume as subplot for context
    if 'Volume' in display_data.columns and not display_data['Volume'].isnull().all():
        # Create range for volume bars
        colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in display_data.iterrows()]

        # Add volume subplot
        fig.add_trace(go.Bar(
            x=display_data.index,
            y=display_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5,
            yaxis='y2'  # Use secondary y-axis
        ))

        # Update layout for dual y-axis
        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

    st.plotly_chart(fig, use_container_width=True)
    
    # Create tabs for additional indicators to improve organization
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Momentum Score"])
    with tab1:
        # Render RSI chart with optimizations
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#7b1fa2', width=2)
        ))

        # Add reference lines for overbought/oversold conditions
        rsi_fig.add_trace(go.Scatter(
            x=[display_data.index[0], display_data.index[-1]],
            y=[70, 70],
            mode='lines',
            name='Overbought',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash')
        ))

        rsi_fig.add_trace(go.Scatter(
            x=[display_data.index[0], display_data.index[-1]],
            y=[30, 30],
            mode='lines',
            name='Oversold',
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash')
        ))

        # Add 50 line for neutral reference
        rsi_fig.add_trace(go.Scatter(
            x=[display_data.index[0], display_data.index[-1]],
            y=[50, 50],
            mode='lines',
            name='Neutral',
            line=dict(color='rgba(150, 150, 150, 0.3)', width=1, dash='dot')
        ))

        rsi_fig.update_layout(
            title='RSI (Relative Strength Index)',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            height=300,
            template='plotly_dark',
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )

        st.plotly_chart(rsi_fig, use_container_width=True)

    with tab2:
        # Render MACD chart with optimizations
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#64b5f6', width=2)
        ))

        macd_fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff7043', width=1)
        ))

        # Create colors for histogram - use efficient list comprehension
        macd_colors = ['rgba(0, 255, 0, 0.5)' if val >= 0 else 'rgba(255, 0, 0, 0.5)'
                      for val in display_data['MACD_Histogram']]

        macd_fig.add_trace(go.Bar(
            x=display_data.index,
            y=display_data['MACD_Histogram'],
            name='Histogram',
            marker_color=macd_colors
        ))

        # Add zero line for reference
        macd_fig.add_trace(go.Scatter(
            x=[display_data.index[0], display_data.index[-1]],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='rgba(150, 150, 150, 0.3)', width=1, dash='dot')
        ))

        macd_fig.update_layout(
            title='MACD (Moving Average Convergence Divergence)',
            xaxis_title='Date',
            yaxis_title='MACD Value',
            height=300,
            template='plotly_dark',
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )

        st.plotly_chart(macd_fig, use_container_width=True)

    with tab3:
        # Render Momentum Score chart with optimizations
        momentum_fig = go.Figure()

        # Filter out NaN values for better visualization
        momentum_data = display_data[['Momentum_Score']].dropna()

        if not momentum_data.empty:
            momentum_fig.add_trace(go.Scatter(
                x=momentum_data.index,
                y=momentum_data['Momentum_Score'],
                mode='lines',
                name='Momentum Score',
                line=dict(color='#ffb300', width=2)
            ))

            # Add background color zones for buy/sell/neutral
            momentum_fig.add_shape(
                type="rect",
                x0=momentum_data.index[0],
                y0=60,
                x1=momentum_data.index[-1],
                y1=100,
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )

            momentum_fig.add_shape(
                type="rect",
                x0=momentum_data.index[0],
                y0=0,
                x1=momentum_data.index[-1],
                y1=40,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )

            # Add reference lines
            momentum_fig.add_trace(go.Scatter(
                x=[momentum_data.index[0], momentum_data.index[-1]],
                y=[50, 50],
                mode='lines',
                name='Neutral',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash')
            ))

            momentum_fig.add_trace(go.Scatter(
                x=[momentum_data.index[0], momentum_data.index[-1]],
                y=[60, 60],
                mode='lines',
                name='Buy Threshold',
                line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash')
            ))

            momentum_fig.add_trace(go.Scatter(
                x=[momentum_data.index[0], momentum_data.index[-1]],
                y=[40, 40],
                mode='lines',
                name='Sell Threshold',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash')
            ))

            # Add current momentum score as annotation
            latest_score = momentum_data['Momentum_Score'].iloc[-1]
            score_color = 'green' if latest_score > 60 else 'red' if latest_score < 40 else 'white'

            momentum_fig.add_annotation(
                x=momentum_data.index[-1],
                y=latest_score,
                text=f"Score: {latest_score:.1f}",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=-40,
                font=dict(color=score_color)
            )

            momentum_fig.update_layout(
                title='Technical Momentum Score',
                xaxis_title='Date',
                yaxis_title='Score (0-100)',
                height=300,
                template='plotly_dark',
                yaxis=dict(range=[0, 100]),
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=False
            )

            st.plotly_chart(momentum_fig, use_container_width=True)

            # Add interpretation of the current momentum
            current_score = momentum_data['Momentum_Score'].iloc[-1]
            if current_score > 60:
                st.success(f"Strong bullish momentum detected (Score: {current_score:.1f})")
            elif current_score < 40:
                st.error(f"Strong bearish momentum detected (Score: {current_score:.1f})")
            else:
                st.info(f"Neutral momentum (Score: {current_score:.1f})")
        else:
            st.warning("Insufficient data to calculate momentum score.")
            
def display_strength_data(ticker, data, momentum_score, signals, session_state, unique_id):
    """Displays strength data and indicator values with price information."""
    col1, col2, col3, col4, col5 = st.columns([2, 3, 3, 2, 2])
    
    # Extract price information - assuming this is in your data
    # If not available, you'll need to fetch it
    current_price = data["Close"].iloc[-1]
    day_change_pct = data["Close"].pct_change().iloc[-1] * 100
    
    # Format day change with color based on positive/negative
    if day_change_pct > 0:
        day_change_html = f'<span style="color:#4CAF50">+{day_change_pct:.2f}%</span>'
    elif day_change_pct < 0:
        day_change_html = f'<span style="color:#f44336">{day_change_pct:.2f}%</span>'
    else:
        day_change_html = f'<span>{day_change_pct:.2f}%</span>'
    
    with col1:
        st.write(f"**{ticker}**")
    
    with col2:
        # Display key signals
        signal_html = ""
        for key in ["MACD", "RSI", "Supertrend"]:
            if key in signals:
                # Convert signal text to match CSS class naming
                signal_text = signals[key]['signal'].replace(" ", "-").lower()
                
                # Create proper class name that matches your CSS
                if signal_text in ["strong-buy", "buy", "weak-buy", "weak-sell", "sell", "strong-sell", "neutral"]:
                    signal_class = f"indicator-{signal_text}"
                else:
                    # Fallback if signal doesn't match expected format
                    signal_class = "indicator-neutral"
                    
                signal_html += f'<span class="indicator-pill {signal_class}">{key}: {signals[key]["value"]}</span> '
        
        st.markdown(signal_html, unsafe_allow_html=True)
    
    with col3:
        st.write(f"Momentum: **{momentum_score:.2f}**")
        
        # Format the overall momentum signal properly
        direction = signals["Overall Momentum"]["signal"]
        
        # Format direction to match CSS class naming
        direction_class = direction.replace(" ", "-").lower()
        
        # Create the HTML with proper classes
        direction_html = f'<span class="indicator-pill indicator-{direction_class}">Signal: {direction}</span>'
        
        st.markdown(direction_html, unsafe_allow_html=True)
    
    with col4:
        # Display current price and day change
        price_html = f"""
        <div>₹{current_price:.2f}</div>
        <div>{day_change_html}</div>
        """
        st.markdown(price_html, unsafe_allow_html=True)
    
    with col5:
        # Use unique_id to ensure unique key for each button
        button_key = f"graph_{ticker}_{unique_id}_{int(time.time() * 1000)}"
        if st.button("Show Graph", key=button_key, on_click=update_state, args=(ticker, data, session_state)):
            pass  # Callback handles the state update

def update_state(ticker, data, session_state):
    """Callback function to update session state."""
    session_state.selected_ticker = ticker
    session_state.selected_data = data
    session_state.show_graph = True
    time.sleep(0.5)  # Delay for state persistence

def close_graph(session_state):
    """Callback function to close the graph."""
    session_state.show_graph = False
    session_state.selected_ticker = None
    session_state.selected_data = None

# Main app function
def main():
    # Initialize session state
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = None
    if 'show_graph' not in st.session_state:
        st.session_state.show_graph = False
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'scan_id' not in st.session_state:
        st.session_state.scan_id = None
    
    st.title("Advanced Technical Momentum Scanner")
    
    # Sidebar settings
    st.sidebar.header("Scan Settings")
    index_list = list(INDEX_CONSTITUENTS.keys())
    index_choice = st.sidebar.selectbox("Select Index:", index_list)
    ticker_list = get_constituents(index_choice)
    
    
    # Period selection
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=2)
    
    # Interval selection
    interval = st.sidebar.selectbox("Time Interval", ["15m", "1h", "1d", "5d", "1wk", "1mo"], index=2)
    
    # Alert settings
    st.sidebar.header("Alert Settings")
    
    momentum_threshold = st.sidebar.slider("Momentum Threshold", 10, 40, 20, 5)
    rsi_threshold_high = st.sidebar.slider("RSI Overbought", 60, 90, 70, 5)
    rsi_threshold_low = st.sidebar.slider("RSI Oversold", 10, 40, 30, 5)
    
    # Email alerts
    enable_email_alerts = st.sidebar.checkbox("Enable Email Alerts")
    
    to_email = ""
    from_email = ""
    password = ""
    
    if enable_email_alerts:
        to_email = st.sidebar.text_input("Your Email")
        from_email = st.sidebar.text_input("Gmail Account")
        password = st.sidebar.text_input("App Password", type="password", 
                                        help="Use an App Password if you have 2FA enabled")
    
    # Clear alerts button
    if "alerts" in st.session_state and st.session_state.alerts:
        if st.sidebar.button("Clear Alerts"):
            st.session_state.alerts = []
            st.sidebar.success("Alerts cleared!")
    
    # Adjust indicator weights
    weights = add_weight_adjustment_section()
    
    # Add a container for the graph that's shown regardless of other state
    graph_container = st.container()
    
    # Run scanner
    run_scanner = st.button("Run Technical Scan")
    
    if run_scanner:
        progress_bar = st.progress(0)
        results = []
        
        # Generate a unique ID for this scan to ensure unique button keys
        scan_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.scan_id = scan_id
        
        for i, ticker in enumerate(ticker_list):
            progress_bar.progress((i + 1) / len(ticker_list))
            
            data = get_asset_data(ticker, period, interval)
            if data is not None and not data.empty:
                try:
                    data_with_indicators = calculate_advanced_indicators(data)
                    if data_with_indicators is not None and not data_with_indicators.empty:
                        momentum_score, signals = calculate_momentum_score(data_with_indicators, weights)
                        results.append({
                            "ticker": ticker,
                            "data": data_with_indicators,
                            "momentum_score": momentum_score,
                            "signals": signals
                        })
                        
                        # Check for alerts
                        check_alerts(ticker, data_with_indicators, momentum_score, signals,
                                   momentum_threshold, rsi_threshold_high, rsi_threshold_low,
                                   enable_email_alerts, to_email, from_email, password)
                except Exception as e:
                    st.error(f"Error processing {ticker}: {str(e)}")
        
        progress_bar.empty()
        
        if results:
            # Store results in session state
            st.session_state.results = results
            
            # Sort results by momentum score (descending)
            results.sort(key=lambda x: x["momentum_score"], reverse=True)
            
            # Display strongest bullish signals
            st.subheader("Strongest Bullish Signals")
            bullish_results = [r for r in results if r["momentum_score"] > 50]
            if bullish_results:
                for result in bullish_results[:5]:  # Show top 5
                    display_strength_data(result["ticker"], result["data"], 
                                        result["momentum_score"], result["signals"], 
                                        st.session_state, scan_id)
            else:
                st.info("No strong bullish signals detected.")
            
            # Display strongest bearish signals
            st.subheader("Strongest Bearish Signals")
            bearish_results = [r for r in results if r["momentum_score"] < 50]
            if bearish_results:
                bearish_sorted = sorted(bearish_results, key=lambda x: x["momentum_score"])
                for result in bearish_sorted[:5]:  # Show top 5
                    display_strength_data(result["ticker"], result["data"], 
                                        result["momentum_score"], result["signals"], 
                                        st.session_state, scan_id)
            else:
                st.info("No strong bearish signals detected.")
            
            # Display all results in a table
            st.subheader("All Results")
            table_data = []
            for result in results:
                signal = result["signals"].get("Overall Momentum", {}).get("signal", "neutral")
                rsi = result["signals"].get("RSI", {}).get("value", "N/A")
                macd = result["signals"].get("MACD", {}).get("value", "N/A")
                supertrend = result["signals"].get("Supertrend", {}).get("value", "N/A")
                table_data.append({
                    "Ticker": result["ticker"],
                    "Momentum Score": f"{result['momentum_score']:.2f}",
                    "Signal": signal.upper(),
                    "RSI": rsi,
                    "MACD": macd,
                    "Supertrend": supertrend
                })
            
            st.dataframe(pd.DataFrame(table_data).set_index("Ticker"), use_container_width=True)
        else:
            st.warning("No results found. Check your ticker symbols and try again.")
    
    # Display graph for selected ticker - this should happen AFTER all other UI elements
    with graph_container:
        
        if st.session_state.show_graph and st.session_state.selected_ticker and st.session_state.selected_data is not None and not st.session_state.selected_data.empty:

            latest_data = st.session_state.selected_data.iloc[-1] # Last row of DataFrame
            latest_open  = latest_data['Open']
            latest_high  = latest_data['High']
            latest_low   = latest_data['Low']
            latest_close = latest_data['Close']
        
            st.header(f"{st.session_state.selected_ticker}")
            st.subheader(f'O: {latest_open:.2f} - H: {latest_high:.2f} - L: {latest_low:.2f} - C: {latest_close:.2f}')
            try:
                plot_advanced_stock_data(st.session_state.selected_ticker, st.session_state.selected_data)
            except Exception as e:
                st.error(f"Graph Error: {str(e)}")
            if st.button("Close Graph", on_click=close_graph, args=(st.session_state,)):
                pass  # Callback handles the state update


if __name__ == "__main__":
    main()
