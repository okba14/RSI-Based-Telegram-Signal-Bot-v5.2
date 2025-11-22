<h1 align="center">🚀 Advanced RSI-Based Telegram Signal Bot v5.2</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Status-Beta-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SMC-Supported-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RSI-MultiTempo-red?style=for-the-badge" />
</p>

---

## 📌 **نظرة عامة**
بوت إشارات تليجرام متقدم يعتمد على مجموعة ضخمة من المؤشرات الفنية، بما في ذلك:

- RSI متعدد الفترات  
- Laguerre RSI  
- SuperTrend  
- Squeeze Momentum  
- Smart Money Concepts (BOS, CHoCH, Order Blocks)  
- ADX, ATR, Bollinger Bands  
- Triple Divergence, Exhaustion, Liquidity Sweep  

يقوم البوت بتحليل جميع أزواج **USDT** على منصة بينانس ديناميكيًا، ويرسل إشارات موثوقة عندما يتحقق تقارب قوي (Confluence).

---

## 🌟 **الميزات**
### 🤖 **المسح الديناميكي للسوق**
- تحميل جميع الأزواج النشطة تلقائيًا  
- التكيف مع الأزواج الجديدة والمزالة  

### 📊 **تقارب المؤشرات**
- استخدام +15 مؤشرًا  
- نظام تسجيل نقاط احترافي للإشارات  

### 🧠 **منطق الإشارات المتقدم**
- Composite Scoring  
- فلترة ADX  
- اقتراحات SL/TP مبنية على ATR  

### 📈 **تحليل SMC**
- BOS / CHoCH  
- Order Blocks  
- الاتجاه الهيكلي  

### ⚙️ **قابلية تخصيص عالية**
- تعديل الإعدادات من `.env` بسهولة  

---

# 🧮 **المؤشرات المدعومة**

### 1️⃣ زخم ومذبذبات
- Multi-Tempo RSI  
- Laguerre RSI  

### 2️⃣ اتجاه
- SuperTrend  
- MAC  
- SMC Trend  

### 3️⃣ تقلب
- Squeeze Momentum  
- Bollinger Bands  

### 4️⃣ هيكل السوق
- ADX  
- BOS / CHoCH  
- Order Blocks  

### 5️⃣ كواشف متقدمة
- Triple Divergence  
- Liquidity Sweep  
- Exhaustion Detector  
- Acceleration Detector  

---

# 🛠️ **التثبيت والإعداد**

## 📥 1. استنساخ المستودع
```bash
git clone https://github.com/okba14/RSI-Based-Telegram-Signal-Bot-v5.2.git
cd RSI-Based-Telegram-Signal-Bot-v5.2
```

## 📦 2. تثبيت المتطلبات: 

```bash
pip install -r requirements.txt
```

⚙️ 3. إنشاء ملف البيئة .env

قم بإنشاء ملف باسم .env داخل مجلد المشروع، ثم ضع فيه الإعدادات التالية:

   ## -- إعدادات API ---
```bash

TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID
BINANCE_API_KEY=YOUR_BINANCE_API_KEY
BINANCE_API_SECRET=YOUR_BINANCE_API_SECRET

# --- إعدادات البوت ---
EXCHANGE_ID=binance
TIMEFRAMES=1h,4h
FETCH_LIMIT=1000
MIN_DATA_POINTS=200

# --- إعدادات المؤشرات ---
# RSI
RSI_SHORT=6
RSI_MID=14
RSI_LONG=28
RSI_SMA_PERIOD=20

# SuperTrend
SUPERTREND_PERIOD=12
SUPERTREND_MULTIPLIER=3.0
SUPERTREND_CHANGE_ATR=true

# Squeeze Momentum
SQUEEZE_BB_LENGTH=20
SQUEEZE_BB_MULT=2.0
SQUEEZE_KC_LENGTH=20
SQUEEZE_KC_MULT=1.5
SQUEEZE_USE_TRUE_RANGE=true

# Laguerre RSI
LAGUERRE_GAMMA=0.6

# MAC
MAC_LENGTH=100
MAC_INCR=10
MAC_FAST=10

# Smart Money Concepts
SMC_SWING_LENGTH=50
SMC_INTERNAL_LENGTH=5
SMC_ORDER_BLOCKS_SIZE=5

# ADX / ATR / Bollinger
ADX_PERIOD=14
ATR_PERIOD=14
BB_PERIOD=20
BB_STD=2

# --- إعدادات كشف الإشارات ---
PREV_TREND_LOOKBACK=20
EXHAUSTION_GAP=15
VOLUME_INCREASE_FACTOR=1.25
MIN_CONFIDENCE_TO_ALERT=0.7

# --- أداء البوت ---
MARKET_SCAN_INTERVAL=300
API_CALL_DELAY=0.5
ALERT_COOLDOWN_MINUTES=15

DEVELOPER_NAME=GUIAR-OQBA
DEVELOPER_EMAIL=techokba@gmail.com
```
▶️ كيفية الاستخدام

بعد إعداد .env، شغّل البوت:
```bash
python main.py
```

سيقوم البوت بـ:

تحميل أسواق USDT

تحليل الأطر الزمنية

اكتشاف BOS / CHoCH

إرسال إشارات إلى تليجرام

---

## ⚠️ إخلاء مسؤولية

هذا البرنامج لأغراض تعليمية فقط.
التداول ينطوي على مخاطر عالية.
لا يعتبر هذا البوت نصيحة مالية.
اختبر دائمًا على بيانات تاريخية (Backtesting).

## ⚠️ ملاحظة هامة

هذه هي **النسخة التجريبية والعامة** من البوت، والتي تعرض قدراته الأساسية فقط.

للوصول إلى **النسخة المتقدمة والكاملة**  
✨ *والتي تتضمن إعدادات محسّنة، ودعم مُفضّل، ومؤشرات حصرية إضافية*  
يرجى التواصل معنا عبر الروابط التالية:

---

## 📦 للحصول على النسخة الكاملة

- 📧 **البريد الإلكتروني:** `techokba@gmail.com` 
- 📩 **تليجرام:** [t.me/maronyo](https://t.me/maronyo)

---

## 👨‍💻 Contact:

- 👤 **DEV:** GUIAR-OQBA  
- 📧 **EMAIL:** `techokba@gmail.com`
- 📱 PHONE: +2136-71-36-04-38


---

  ## 📜 License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
© 2025 **GUIAR OQBA** 🇩🇿  
 with 💻 & ❤️ 

---

Thank you for your support! 🙏
- 📩 **Telegram:** [t.me/maronyo](https://t.me/maronyo)
