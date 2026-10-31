"""
src/chatbot.py
==============
Multi-lingual, Context-Aware Agriculture Chatbot for Tamil Nadu Climate DSS.

Features:
1. ChatGPT-level AI via Gemini REST API (if GEMINI_API_KEY provided or entered in UI)
2. Smart local conversational AI supporting:
   - English
   - Tamil script (தமிழ்)
   - Tanglish (Tamil written in English script: 'mazhai varuma', 'nellu podalama')
3. Multi-intent parser: NEVER returns a simple greeting when a question is asked
4. Context-grounded in Tamil Nadu district forecasts (rainfall, anomalies, hazards)
5. Answers general questions on agriculture, soil, fertilizers, pests, schemes, etc.
"""

import re
import os
import json
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Language & Script Detection
# ---------------------------------------------------------------------------

def is_tamil_script(text: str) -> bool:
    """Check if the text contains Tamil Unicode characters (U+0B80 - U+0BFF)."""
    return any('\u0B80' <= char <= '\u0BFF' for char in text)


def is_tanglish(text: str) -> bool:
    """Check if text is Tanglish (phonetic Tamil in Latin script)."""
    tanglish_keywords = [
        "mazhai", "malai", "varuma", "varum", "varathu", "nellu", "nel",
        "vivasayam", "vivasaayi", "thanni", "neer", "uramm", "uram",
        "marundhu", "marunthu", "poochi", "saagubadi", "sagubadi",
        "podalama", "nadalama", "eppadi", "eppo", "yepdi", "naalaiku",
        "nalaiku", "indru", "iniki", "veppam", "kaathu", "kaalam",
        "solunga", "sollunga", "theriyuma", "paasanam", "pasanam"
    ]
    t = text.lower()
    return any(kw in t for kw in tanglish_keywords)


# ---------------------------------------------------------------------------
# Gemini REST API Call (Zero pip dependencies, uses built-in requests)
# ---------------------------------------------------------------------------

def call_gemini_api(
    user_query: str,
    api_key: str,
    district: str,
    telemetry: Optional[Dict[str, Any]] = None,
    advisory: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Call Google Gemini 1.5 Flash REST API if key is available."""
    if not requests or not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Build rich agricultural & climate system context
    t_info = ""
    if telemetry:
        t_info = (
            f"\nCURRENT DISTRICT CONTEXT for {district}:\n"
            f"- Forecast Rainfall: {telemetry.get('rain_mm', 'N/A')} mm\n"
            f"- Rain Anomaly vs Normal: {telemetry.get('rain_anomaly', 'N/A')} mm\n"
            f"- Dominant Hazard Status: {telemetry.get('hazard_status', 'Normal')}\n"
            f"- ENSO Phase (Niño 3.4): {telemetry.get('enso', 0.0)} °C\n"
            f"- Indian Ocean Dipole (DMI): {telemetry.get('iod', 0.0)}\n"
        )
    if advisory:
        t_info += (
            f"- Agro-Climatic Zone: {advisory.get('agro_climatic_zone', 'N/A')}\n"
            f"- Sowing Strategy: {advisory.get('sowing_strategy', '')}\n"
            f"- Irrigation Strategy: {advisory.get('irrigation_strategy', '')}\n"
        )

    system_prompt = (
        "You are 'Uzhavan AI' (உழவன் ஏஐ), an expert agricultural scientist, climate advisor, "
        "and farmer decision support assistant specializing in Tamil Nadu agriculture.\n"
        "You can converse fluently in English, pure Tamil (தமிழ்), or Tanglish (Tamil in English script).\n"
        "Always respond in the SAME language or script that the user used:\n"
        "- If user asks in Tamil (தமிழ்), reply in fluent Tamil (தமிழ்) with clear farming guidance.\n"
        "- If user asks in Tanglish, reply in Tanglish or clear bilingual Tamil/English.\n"
        "- If user asks in English, reply in English.\n"
        "Ground your advice in actual agronomy, TNAU recommendations, and the current district forecast data provided below.\n"
        "Be warm, encouraging, practical, and direct. Use markdown formatting, bullet points, and emojis."
        f"{t_info}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_query}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 800
        }
    }

    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=12
        )
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Smart Local Conversational Intelligence (No API Key Required)
# ---------------------------------------------------------------------------

def clean_query(text: str) -> str:
    """Strip common greetings from query so substantive questions are detected."""
    t = text.lower()
    # Remove leading greeting prefixes like "hi ", "hello ", "vanakkam ", etc.
    t = re.sub(r"^(hi|hello|hey|vanakkam|namaste|good\s+morning|good\s+afternoon|good\s+evening)\b[\s,\.!]*", "", t).strip()
    return t


def answer_weather_query(district: str, telemetry: Optional[Dict[str, Any]], is_tamil: bool, is_tan: bool) -> str:
    """Handle rainfall / weather forecast questions."""
    rain_mm = telemetry.get("rain_mm", 120.0) if telemetry else 120.0
    rain_anom = telemetry.get("rain_anomaly", 15.0) if telemetry else 15.0
    hazard = telemetry.get("hazard_status", "Low") if telemetry else "Low"

    anom_text = f"+{rain_anom:.1f}" if rain_anom >= 0 else f"{rain_anom:.1f}"

    if is_tamil:
        return (
            f"🌧️ **{district} மாவட்ட வானிலை மற்றும் மழை முன்னறிவிப்பு:**\n\n"
            f"- **எதிர்பார்க்கப்படும் மழைப்பொழிவு:** சுமார் **{rain_mm:.1f} மி.மீ**\n"
            f"- **இயல்பு நிலையை விட வேறுபாடு:** **{anom_text} மி.மீ** ({'அதிக மழை' if rain_anom > 20 else ('குறைவான மழை' if rain_anom < -20 else 'சராசரி மழை')})\n"
            f"- **பாதிப்பு அபாய நிலை:** **{hazard}**\n\n"
            f"💡 **விவசாயிகளுக்கான வழிகாட்டல்:**\n"
            f"• {'அதிக மழை வாய்ப்பு உள்ளதால், வயல்களில் வடிகால் கால்வாய்களை உடனே தூர்வாரி தயார் நிலையில் வையுங்கள்.' if rain_anom > 20 else 'மழைப்பொழிவு குறைவாகவோ அல்லது மிதமாகவோ இருக்கும் என கணிக்கப்பட்டுள்ளதால், நீர் மேலாண்மையில் (AWD/சொட்டு நீர்) கவனம் செலுத்துங்கள்.'}\n"
            f"• வானிலை மாற்றங்களை தொடர்ந்து கண்காணிக்கவும்."
        )
    elif is_tan:
        return (
            f"🌧️ **{district} Weather & Mazhai Mun-arivippu:**\n\n"
            f"- **Expected Rainfall:** **{rain_mm:.1f} mm**\n"
            f"- **Normal-oda compare panna:** **{anom_text} mm** ({'Adhiga mazhai' if rain_anom > 20 else 'Sari-aana mazhai / Deficit'})\n"
            f"- **Hazard Status:** **{hazard}**\n\n"
            f"💡 **Farming Tip:**\n"
            f"• {'Mazhai nallave irukku, vadikaal (drainage) vazhi-a clear pannunga!' if rain_anom > 20 else 'Mazhai kami-a irundha, AWD or Drip irrigation use pannunga.'}\n"
            f"• Next 2-3 weeks forecast-ah follow pannunga."
        )
    else:
        return (
            f"🌧️ **Weather & Rainfall Outlook for {district}:**\n\n"
            f"- **Forecasted Monthly Rainfall:** **{rain_mm:.1f} mm**\n"
            f"- **Anomaly vs Normal:** **{anom_text} mm** ({'Above Normal / Surplus' if rain_anom > 20 else ('Below Normal / Deficit' if rain_anom < -20 else 'Near Normal')})\n"
            f"- **Hazard Risk Level:** **{hazard}**\n\n"
            f"💡 **Agronomic Advisory:**\n"
            f"• {'Heavy rain anticipated. Clear field drainage channels to prevent waterlogging around root zones.' if rain_anom > 20 else 'Rainfall expected to be moderate/deficit. Adopt Alternate Wetting & Drying (AWD) or micro-irrigation to conserve soil moisture.'}\n"
            f"• Avoid applying chemical fertilizers immediately before predicted rain peaks to prevent nutrient runoff."
        )


def answer_paddy_query(district: str, telemetry: Optional[Dict[str, Any]], advisory: Optional[Dict[str, Any]], is_tamil: bool, is_tan: bool) -> str:
    """Handle paddy cultivation queries."""
    rain_anom = telemetry.get("rain_anomaly", 0.0) if telemetry else 0.0
    hazard = telemetry.get("hazard_status", "Low") if telemetry else "Low"

    if is_tamil:
        return (
            f"🌾 **{district} மாவட்டத்திற்கான நெல் சாகுபடி ஆலோசனை:**\n\n"
            f"நீங்கள் இப்போது நெல் பயிரிட திட்டமிட்டால்:\n\n"
            f"1. **பரிந்துரைக்கப்படும் நெல் ரகங்கள்:**\n"
            f"   - **ADT 37 / TKM 13** — நடுத்தர கால ரகம் (115–120 நாட்கள்)\n"
            f"   - **CO 51** — குறுகிய கால வறட்சியை தாங்கும் ரகம் (105 நாட்கள்)\n"
            f"   - **Swarna-Sub1 / CR 1009 Sub1** — வெள்ளம் / அதிக நீர் தேங்குவதை தாங்கும் ரகம் (2-3 வாரம் நீர் மூழ்கினாலும் அழுகாது)\n"
            f"   - **ASD 16** — கடற்கரை பகுதி உப்பு தன்மையை தாங்கும் ரகம் ({district} கடற்கரைக்கு ஏற்றது)\n\n"
            f"2. **நடவு முறை:**\n"
            f"   - நாற்றங்கால் பாத்திகளை மேட்டுப்பாத்திகளாக அமைக்கவும்.\n"
            f"   - விதைப்பதற்கு முன் 1 கிலோ விதைக்கு 2 கிராம் கார்பென்டாசிம் மருந்து கலந்து விதை நேர்த்தி செய்யவும்.\n\n"
            f"3. **தண்ணீர் மேலாண்மை:**\n"
            f"   - தூர்கட்டும் பருவம் வரை 2 செ.மீ தண்ணீர் நிறுத்தி, பின் காய்ச்சலும் பாய்ச்சலுமாக (AWD) நீர் பாய்ச்சினால் 25% தண்ணீர் மிச்சமாகும்."
        )
    elif is_tan:
        return (
            f"🌾 **{district} Nellu Saagubadi (Paddy Cultivation) Tips:**\n\n"
            f"Neenga nellu poda plan pannirundha:\n\n"
            f"1. **Best Nellu Ragaangal (Varieties):**\n"
            f"   - **CO 51** — 105 days, water kammiya theva padum.\n"
            f"   - **ADT 37 / TKM 13** — 115-120 days, high yield.\n"
            f"   - **Swarna-Sub1 / CR 1009 Sub1** — Mazhai vandhu thanni thenginalum thalaikattum (Flood resistant).\n"
            f"   - **ASD 16** — Coastal {district} areas-ku super.\n\n"
            f"2. **Thanni Manage panna:**\n"
            f"   - Always AWD (Kaaichalum Paaichalum) method use pannunga, 25% thanni save aagum.\n"
            f"   - Naatru nadum podhu seed treatment kandippa pannunga."
        )
    else:
        return (
            f"🌾 **Paddy Cultivation & Planting Advisory for {district}:**\n\n"
            f"If you are planning to sow or transplant paddy:\n\n"
            f"1. **Recommended Cultivars for {district}:**\n"
            f"   - **CO 51** — Short-duration (105 days), highly resilient to moisture stress\n"
            f"   - **ADT 37 / TKM 13** — Medium-duration (115–120 days), optimal yield potential\n"
            f"   - **Swarna-Sub1 / CR 1009 Sub1** — Submergence-tolerant; survives 14–18 days submerged in flood-prone fields\n"
            f"   - **ASD 16** — Salt-tolerant cultivar suitable for coastal belts\n\n"
            f"2. **Nursery & Sowing Strategy:**\n"
            f"   - Raise seedlings on raised nursery beds with 10 cm clearance to prevent early submergence rot\n"
            f"   - Perform seed treatment with *Pseudomonas fluorescens* (10 g/kg) to guard against blast\n\n"
            f"3. **Water Management (AWD):**\n"
            f"   - Implement Alternate Wetting & Drying: re-irrigate only when water recedes 15 cm below soil level."
        )


def answer_combined_paddy_and_rain(district: str, telemetry: Optional[Dict[str, Any]], advisory: Optional[Dict[str, Any]], is_tamil: bool, is_tan: bool) -> str:
    """Specifically handles 'hi i planned to grow paddy will it rain tommorow'."""
    rain_mm = telemetry.get("rain_mm", 140.0) if telemetry else 140.0
    rain_anom = telemetry.get("rain_anomaly", 20.0) if telemetry else 20.0
    hazard = telemetry.get("hazard_status", "Low") if telemetry else "Low"

    anom_sign = f"+{rain_anom:.1f}" if rain_anom >= 0 else f"{rain_anom:.1f}"

    if is_tamil:
        return (
            f"🌾 **{district} - நெல் சாகுபடி & மழை முன்னறிவிப்பு விளக்கம்:**\n\n"
            f"வணக்கம்! நீங்கள் {district} மாவட்டத்தில் நெல் பயிரிட திட்டமிட்டுள்ளீர்கள், மழை பற்றிய உங்கள் கேள்விக்கு:\n\n"
            f"🌧️ **மழை நிலவரம்:**\n"
            f"- தற்போதைய காலநிலை மாதிரியின்படி, இந்த மாதத்தில் சுமார் **{rain_mm:.1f} மி.மீ** மழை ({anom_sign} மி.மீ இயல்பை விட) கணிக்கப்பட்டுள்ளது.\n"
            f"- தற்போதைய அபாய நிலை: **{hazard}**.\n\n"
            f"🌱 **நெல் நடவு செய்யலாமா?**\n"
            f"• **ஆம், தாராளமாக திட்டமிடலாம்!** ஆனால் மழைநீரை மட்டுமே நம்பாமல், பாசன ஆதாரத்தையும் உறுதிப்படுத்திக் கொள்ளுங்கள்.\n"
            f"• அதிக மழை எதிர்பார்க்கப்பட்டால், நீர் தேங்குவதை தாங்கும் **Swarna-Sub1** அல்லது **CR 1009 Sub1** ரகங்களை தேர்வு செய்யவும்.\n"
            f"• கடற்கரை சார்ந்த பகுதியாக இருப்பின் **ASD 16** சிறந்த ரகமாகும்.\n"
            f"• நாற்று நடும் முன் வயல் வரப்புகளை உயர்த்தி, அதிகப்படியான நீர் வெளியேற வடிகால் வாய்க்காலை சரிசெய்யவும்.\n\n"
            f"உங்களுக்கு குறிப்பிட்ட ரகங்கள் அல்லது பூச்சி மேலாண்மை பற்றி ஏதேனும் சந்தேகம் உள்ளதா?"
        )
    elif is_tan:
        return (
            f"🌾 **{district} - Nellu Saagubadi & Mazhai Information:**\n\n"
            f"Vanakkam! Neenga {district}-la nellu poda plan pannirukenga, mazhai pathina details:\n\n"
            f"🌧️ **Mazhai Forecast:**\n"
            f"- Climate forecast-padi **{rain_mm:.1f} mm** ({anom_sign} mm vs normal) mazhai expect pannapadudhu.\n"
            f"- Risk level: **{hazard}**.\n\n"
            f"🌱 **Nellu Saagubadi Tips:**\n"
            f"• **Yes, neenga tharalama proceed pannalam!** Aana borewell/canal irrigation backup vachukonga.\n"
            f"• Mazhai adhigama irundha flood-tolerant varieties like **Swarna-Sub1** or **CR 1009 Sub1** podunga.\n"
            f"• Water logging aagama iruka drainage clear pannitu nursery transplant pannunga.\n"
            f"• Kaaichalum paaichaluma (AWD) thanni paasanam pannunga."
        )
    else:
        return (
            f"🌾 **Paddy Cultivation & Rainfall Assessment for {district}:**\n\n"
            f"Great decision to plan ahead! Here is the hydro-meteorological assessment for **{district}**:\n\n"
            f"🌧️ **Rainfall Forecast Outlook:**\n"
            f"- **Expected Monthly Rainfall:** **{rain_mm:.1f} mm**\n"
            f"- **Rainfall Anomaly vs Baseline Normal:** **{anom_sign} mm**\n"
            f"- **Overall Hazard Status:** **{hazard}**\n\n"
            f"🌱 **Actionable Guidance for Growing Paddy:**\n"
            f"1. **Is it favorable to plant now?** Yes! The hydro-climatic conditions in {district} support paddy planting, provided irrigation scheduling is aligned with forecast events.\n"
            f"2. **Cultivar Choice:**\n"
            f"   - If rainfall is in surplus / flood risk: Opt for **Swarna-Sub1** or **CR 1009 Sub1** (submergence-tolerant for up to 18 days).\n"
            f"   - For normal/moderate rain: Standard high-yielding cultivars like **ADT 37**, **CO 51**, or **TKM 13**.\n"
            f"   - For coastal {district}: Saline-tolerant **ASD 16**.\n"
            f"3. **Precaution:** Clear field bund drainage channels to manage unexpected monsoon downpours and avoid seedling submergence."
        )


def answer_general_agri_query(query: str, district: str, is_tamil: bool, is_tan: bool) -> str:
    """Answers general agricultural questions (pests, fertilizers, organic, irrigation, etc.)."""
    q = query.lower()

    # 1. Pests / Diseases
    if any(k in q for k in ["pest", "disease", "poochi", "marundhu", "blast", "borer", "leaf folder", "kuruthu"]):
        if is_tamil:
            return (
                f"🐛 **பூச்சி மற்றும் நோய் மேலாண்மை (IPM வழிகாட்டல்):**\n\n"
                f"• **குருத்துப்பூச்சி (Stem Borer):** வேப்பங்கொட்டை கரைசல் (NSKE 5%) அல்லது கார்டாப் ஹைட்ரோகுளோரைடு (18.75 கிலோ/ஹெக்டேர்).\n"
                f"• **இலைமடக்கு புழு (Leaf Folder):** டிரைக்கோடெர்மா ஒட்டுண்ணி அட்டைகளை ஒரு ஏக்கருக்கு 2 வீதம் வயலில் கட்டவும்.\n"
                f"• **குலை நோய் (Blast):** டிரைசைக்ளோசோல் 75 WP (0.6 கிராம்/லிட்டர்) தெளிக்கவும்.\n"
                f"• **பரிந்துரை:** இயற்கை முறையில் முதலில் வேப்பெண்ணெய் அல்லது பஞ்சகவ்யா தெளித்து கட்டுப்படுத்துவது நல்லது."
            )
        else:
            return (
                f"🐛 **Pest & Disease Control (Integrated Pest Management):**\n\n"
                f"• **Stem Borer:** Apply Neem Seed Kernel Extract (NSKE 5%) or Cartap Hydrochloride 4G.\n"
                f"• **Leaf Folder:** Release *Trichogramma* parasitoid egg cards (2 cards/acre) at weekly intervals.\n"
                f"• **Paddy Blast (Magnaporthe oryzae):** Spray Tricyclazole 75 WP @ 0.6 g/L or Pseudomonos fluorescens foliar spray.\n"
                f"• **Bio-control First:** Prefer organic bio-pesticides before turning to synthetic chemical sprays."
            )

    # 2. Organic Farming / Panchagavya
    if any(k in q for k in ["organic", "iyarkai", "panchagavya", "jeevamrutham", "vermicompost", "manure"]):
        if is_tamil:
            return (
                f"🌱 **இயற்கை வேளாண்மை மற்றும் பஞ்சகவ்யா வழிகாட்டல்:**\n\n"
                f"• **பஞ்சகவ்யா தயாரிப்பு:** பசுஞ்சாணம், கோமியம், பால், தயிர், நெய், நாட்டுச்சர்க்கரை, இளநீர், கனிந்த வாழைப்பழம் ஆகியவற்றை சேர்த்து 21 நாட்கள் நொதிக்க வைக்கவும்.\n"
                f"• **பயன்பாடு:** 3% கரைசல் (100 லிட்டர் தண்ணீருக்கு 3 லிட்டர் பஞ்சகவ்யா) தெளித்தால் பயிரின் நோய் எதிர்ப்பு சக்தி கூடும்.\n"
                f"• **ஜீவாமிர்தம்:** மண் நுண்ணுயிரிகளை பெருக்கி வேர் வளர்ச்சியை விரைவுபடுத்தும்."
            )
        else:
            return (
                f"🌱 **Organic Farming & Bio-Enhancers:**\n\n"
                f"• **Panchagavya:** Traditional 9-ingredient bio-enhancer (cow dung, urine, milk, curd, ghee, jaggery, tender coconut, banana). Use as a 3% foliar spray.\n"
                f"• **Jeevamrutham:** Fermented cow dung-urine culture with pulse flour; drastically multiplies beneficial soil microbes.\n"
                f"• **Vermicompost:** Apply 2.5 tonnes/ha to enhance soil moisture retention and organic carbon."
            )

    # 3. Irrigation / AWD / Drip
    if any(k in q for k in ["irrigation", "water", "thanni", "awd", "drip", "pasanam"]):
        if is_tamil:
            return (
                f"💧 **பாசனம் மற்றும் நீர் சேமிப்பு தொழில்நுட்பம்:**\n\n"
                f"• **காய்ச்சலும் பாய்ச்சலும் முறை (AWD):** மண்ணில் 15 செ.மீ துளையிடப்பட்ட பிளாஸ்டிக் குழாயை வைத்து, நீர் தரைமட்டத்திற்கு கீழே 15 செ.மீ இறங்கிய பின் மீண்டும் நீர் பாய்ச்ச வேண்டும்.\n"
                f"• **பலன்:** 25-30% தண்ணீர் சேமிப்புடன், மீத்தேன் வாயு வெளியேற்றமும் குறைகிறது.\n"
                f"• **சொட்டு நீர் பாசனம்:** கரும்பு, பருத்தி, மக்காச்சோளம் மற்றும் தோட்டக்கலை பயிர்களுக்கு 45-90% மானியத்தில் அரசு வழங்குகிறது."
            )
        else:
            return (
                f"💧 **Water Management & Irrigation Efficiency:**\n\n"
                f"• **Alternate Wetting & Drying (AWD):** Monitor water level using a perforated PVC pipe; re-irrigate only when water drops 15 cm below soil level.\n"
                f"• **Water Savings:** 25–30% without any yield penalty.\n"
                f"• **Drip Irrigation:** Highly recommended for cotton, sugarcane, maize, and orchards; TN government provides 45–90% subsidy."
            )

    # 4. Government Schemes / PM-Kisan / PMFBY
    if any(k in q for k in ["scheme", "subsidy", "maniyam", "pmfby", "kisan", "insurance", "kaappeedu"]):
        if is_tamil:
            return (
                f"🏛️ **தமிழ்நாடு விவசாய அரசு திட்டங்கள் மற்றும் மானியங்கள்:**\n\n"
                f"• **PMFBY (பயிர் காப்பீட்டுத் திட்டம்):** வறட்சி, வெள்ளம் போன்ற இயற்கை பேரிடரால் ஏற்படும் பயிர் இழப்பிற்கு முழு காப்பீடு.\n"
                f"• **PM-KISAN:** ஆண்டிற்கு ₹6,000 மூன்று தவணைகளில் நேரடியாக வங்கிக் கணக்கில்.\n"
                f"• **நுண்ணீர் பாசன மானியம்:** சிறு/குறு விவசாயிகளுக்கு 100% மற்றும் இதர விவசாயிகளுக்கு 75% வரை சொட்டு நீர் மானியம்.\n"
                f"• **விண்ணப்பிக்க:** அருகில் உள்ள வட்டார வேளாண்மை விரிவாக்க மையம் அல்லது உழவன் செயலியை (Uzhavan App) அணுகவும்."
            )
        else:
            return (
                f"🏛️ **Key Government Schemes & Subsidies for TN Farmers:**\n\n"
                f"• **PMFBY (Crop Insurance):** Complete risk buffer against drought, cyclone, unseasonal floods, and post-harvest loss.\n"
                f"• **PM-KISAN:** Direct income support of ₹6,000/year in 3 equal installments.\n"
                f"• **Micro-Irrigation Subsidy:** Up to 100% subsidy for small/marginal farmers and 75% for other farmers through TN Horticulture Dept.\n"
                f"• **Access:** Apply via your local Block Agricultural Extension Office or the Tamil Nadu 'Uzhavan' Mobile App."
            )

    # Fallback response
    if is_tamil:
        return (
            f"🌾 **உழவன் ஏஐ (Uzhavan AI) பதில் — {district}:**\n\n"
            f"உங்கள் கேள்வி: *'{query}'*\n\n"
            f"நான் தமிழ்நாடு மாவட்ட காலநிலை அபாயங்கள், மழை முன்னறிவிப்பு, நெல்/பயறு சாகுபடி, உர மேலாண்மை, பூச்சி கட்டுப்பாடு மற்றும் அரசு திட்டங்கள் குறித்து பதிலளிக்க முடியும்.\n\n"
            f"உதாரணமாக கேட்கலாம்:\n"
            f"- *'இப்போது நெல் நடவு செய்யலாமா?'*\n"
            f"- *'அடுத்த மாதம் மழை எப்படி இருக்கும்?'*\n"
            f"- *'வறட்சியை தாங்கும் பயிர்கள் என்ன?'*"
        )
    else:
        return (
            f"🌾 **Uzhavan AI Agriculture Assistant — {district}:**\n\n"
            f"Regarding your query on *'{query}'*:\n\n"
            f"I can assist you with Tamil Nadu climate risk forecasts, crop calendars, pest management (IPM), irrigation techniques (AWD), soil health, and government subsidies.\n\n"
            f"You can also ask:\n"
            f"- *'Will it rain this month in {district}?'*\n"
            f"- *'What are the best drought-resistant paddy varieties?'*\n"
            f"- *'How to apply for crop insurance under PMFBY?'*"
        )


# ---------------------------------------------------------------------------
# Main Router Function
# ---------------------------------------------------------------------------

def get_agriculture_response(
    user_query: str,
    selected_district: Optional[str] = "Chennai",
    advisory: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Main entry point for chatbot responses.

    1. Checks for Gemini API key (from parameter or environment). If present, calls Gemini.
    2. If no key, routes through the smart multi-lingual intent classifier.
    """
    district = selected_district or "Chennai"
    raw_query = user_query.strip()
    q_cleaned = clean_query(raw_query)

    # Check for Gemini API key
    active_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if active_key:
        llm_resp = call_gemini_api(raw_query, active_key, district, telemetry, advisory)
        if llm_resp:
            return llm_resp

    # Local Engine: Detect language
    is_tamil = is_tamil_script(raw_query)
    is_tan = is_tanglish(raw_query)

    # Check if query is just a simple greeting with nothing else
    if not q_cleaned or q_cleaned in ["hi", "hello", "vanakkam", "hey", "namaste"]:
        if is_tamil:
            return (
                f"👋 **வணக்கம்! நான் உங்கள் {district} மாவட்ட வேளாண் மற்றும் காலநிலை உதவியாளர்.**\n\n"
                f"நீங்கள் என்னிடம் கேட்கலாம்:\n"
                f"- 🌧️ மழை முன்னறிவிப்பு & வானிலை நிலவரம்\n"
                f"- 🌾 நெல், பருத்தி, கரும்பு சாகுபடி மற்றும் ரகங்கள்\n"
                f"- 💧 காய்ச்சலும் பாய்ச்சலும் (AWD) பாசன முறை\n"
                f"- 🐛 பயிர் பூச்சி மற்றும் நோய் கட்டுப்பாடு\n"
                f"- 🏛️ பயிர் காப்பீடு & அரசு மானியங்கள்\n\n"
                f"உங்கள் கேள்வியை கீழே தட்டச்சு செய்யவும் அல்லது மைக்ரோஃபோன் மூலம் பேசவும்!"
            )
        elif is_tan:
            return (
                f"👋 **Vanakkam! Welcome to {district} Agri & Climate AI.**\n\n"
                f"Enkitta neenga kekalam:\n"
                f"- 🌧️ Mazhai varuma & Weather forecast\n"
                f"- 🌾 Nellu podalama, best varieties\n"
                f"- 💧 Thanni pasanam & AWD tips\n"
                f"- 🐛 Poochi marundhu & IPM methods\n\n"
                f"Type pannunga or mic use panni pesunga!"
            )
        else:
            return (
                f"👋 **Welcome to the {district} Climate & Agriculture Advisory AI!**\n\n"
                f"I can help you with:\n"
                f"- 🌧️ **Weather & Rainfall Forecasts** for Tamil Nadu districts\n"
                f"- 🌾 **Crop Planting & Cultivar Selection** (Paddy, Millets, Pulses, Cotton)\n"
                f"- 💧 **Water Conservation & AWD Irrigation**\n"
                f"- 🐛 **Pest & Disease Management** (IPM)\n"
                f"- 🏛️ **Government Subsidies & PMFBY Crop Insurance**\n\n"
                f"Ask any question in English, Tamil (தமிழ்), or Tanglish — or tap the 🎙️ mic to speak!"
            )

    # Multi-Intent: Weather + Paddy/Crop planting
    has_rain = any(w in q_cleaned for w in [
        "rain", "weather", "mazhai", "malai", "rainy", "tomorow", "tomorrow", "forecast",
        "மழை", "வானிலை", "மழைய", "புயல்"
    ])
    has_paddy = any(w in q_cleaned for w in [
        "paddy", "rice", "nellu", "nel", "grow", "plant", "sow", "saagubadi", "cultivat",
        "நெல்", "சாகுபடி", "பயிரிட", "நடவு", "விதைக்க"
    ])

    if has_rain and has_paddy:
        return answer_combined_paddy_and_rain(district, telemetry, advisory, is_tamil, is_tan)
    elif has_rain:
        return answer_weather_query(district, telemetry, is_tamil, is_tan)
    elif has_paddy:
        return answer_paddy_query(district, telemetry, advisory, is_tamil, is_tan)
    else:
        return answer_general_agri_query(q_cleaned, district, is_tamil, is_tan)
