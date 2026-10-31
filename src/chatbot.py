"""
src/chatbot.py
==============
Rule-based Agriculture Advisory Chatbot for Tamil Nadu Farmers.

Fully local — no external APIs required.
Covers: crops, ENSO/IOD effects, sowing windows, irrigation, soil health,
district agro-climatic zones, disaster response, and general agronomy.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE — list of (pattern_list, response_template)
# Pattern list uses simple lowercase substring matching.
# ---------------------------------------------------------------------------

AGRI_KB = [
    # ── GREETINGS ────────────────────────────────────────────────────────────
    (
        ["hello", "hi ", "helo", "hay ", "good morning", "good evening",
         "good afternoon", "namaste", "vanakkam", "welcome"],
        (
            "👋 **Vanakkam! Welcome to the TN Agri Advisory Chatbot.**\n\n"
            "I can help you with:\n"
            "- 🌾 **Crop selection** for Tamil Nadu districts\n"
            "- 🌧️ **ENSO / IOD effects** on monsoon rainfall\n"
            "- 💧 **Irrigation & water management** during drought or flood\n"
            "- 🗓️ **Sowing windows** for Kharif (SWM) and Rabi/NEM seasons\n"
            "- 🌱 **Soil health, fertilizer & pest management**\n"
            "- ⚠️ **Flood / drought emergency actions**\n\n"
            "Type your question below — in plain English or Tamil-English mix!"
        )
    ),

    # ── ENSO & EL NIÑO ───────────────────────────────────────────────────────
    (
        ["el nino", "el niño", "enso", "oni", "nino 3.4", "niño"],
        (
            "🌊 **ENSO (El Niño–Southern Oscillation) & Tamil Nadu Rainfall**\n\n"
            "**El Niño (ONI ≥ +0.5 °C):**\n"
            "- Weakens the South-West Monsoon (June–Sept) → **Below-normal rainfall in most of TN**\n"
            "- North-East Monsoon (Oct–Dec) may be enhanced — Thanjavur/Cuddalore receive compensatory rains\n"
            "- Drought risk rises for western & southern districts (Coimbatore, Madurai, Tirunelveli)\n"
            "- **Advisory:** Pre-position seeds of drought-tolerant varieties (CO 51, Samba Mahsuri Sub1); "
            "deepen farm ponds; activate AWD (Alternate Wetting & Drying)\n\n"
            "**La Niña (ONI ≤ -0.5 °C):**\n"
            "- Strengthens SWM → **Above-normal rainfall; flood risk in delta districts**\n"
            "- NEM rainfall may be erratic\n"
            "- **Advisory:** Ensure drainage channels are cleared; use raised-bed cultivation; "
            "plant submergence-tolerant varieties (Swarna-Sub1, CR 1009 Sub1)\n\n"
            "**Neutral ENSO:** Near-normal monsoon; follow standard agronomic calendar."
        )
    ),

    # ── IOD / INDIAN OCEAN DIPOLE ───────────────────────────────────────────
    (
        ["iod", "indian ocean dipole", "dmi", "dipole"],
        (
            "🌏 **Indian Ocean Dipole (IOD) & Tamil Nadu**\n\n"
            "The IOD measures sea surface temperature difference between the Western (Arabian Sea) "
            "and Eastern (Bay of Bengal) Indian Ocean.\n\n"
            "**Positive IOD (DMI ≥ +0.2):**\n"
            "- Suppresses rainfall over SE India during SWM\n"
            "- **But enhances NEM** — Bay of Bengal feeds extra moisture to TN coast\n"
            "- Cauvery Delta (Thanjavur, Cuddalore) benefits from Oct–Dec NEM enhancement\n\n"
            "**Negative IOD (DMI ≤ -0.2):**\n"
            "- May reduce NEM rainfall → drought risk for Cauvery dependant districts\n"
            "- Coimbatore & Salem see reduced irrigation reservoir inflows\n\n"
            "**Rule of thumb:** Positive IOD + La Niña = heavy NEM rains. "
            "Positive IOD + El Niño = SWM deficit but NEM partly rescued."
        )
    ),

    # ── MONSOON SEASONS ──────────────────────────────────────────────────────
    (
        ["north east monsoon", "nem", "northeast", "winter rain", "retreating",
         "south west monsoon", "swm", "kharif monsoon", "southwest"],
        (
            "🌧️ **Tamil Nadu Monsoon Seasons**\n\n"
            "**South-West Monsoon (SWM) — June to September:**\n"
            "- Arrives in TN by early June; weakest in Chennai, stronger in Nilgiris/Coimbatore\n"
            "- Western Ghats act as a barrier → **rain-shadow on eastern TN**\n"
            "- Kharif (Sornavari + Samba) crop season\n\n"
            "**North-East Monsoon (NEM) — October to December:**\n"
            "- **TN's primary rainfall season** — 60% of annual rainfall in coastal/delta districts\n"
            "- Drives Cauvery inflows → critical for Thanjavur, Cuddalore paddy\n"
            "- Oct: onset; Nov: peak; Dec: withdrawal\n\n"
            "**Rabi / Pishanam (Jan–Mar):** Minimal rain; irrigation-dependent crops only\n"
            "**Summer (Apr–May):** Pre-monsoon showers; mango, groundnut planting in some zones"
        )
    ),

    # ── PADDY / RICE ─────────────────────────────────────────────────────────
    (
        ["paddy", "rice", "adt 37", "cr 1009", "swarna", "samba mahsuri",
         "kuruvai", "samba", "thaladi", "co 51", "asd 16"],
        (
            "🌾 **Tamil Nadu Paddy (Rice) Varieties & Recommendations**\n\n"
            "**Season Mapping:**\n"
            "| Season | Period | Districts |\n"
            "|--------|--------|-----------|\n"
            "| Kuruvai | Jun–Jul planting → Sep harvest | Thanjavur, Cuddalore, Tiruchy |\n"
            "| Samba | Aug–Sep planting → Jan harvest | All delta, Salem |\n"
            "| Thaladi | Nov–Dec planting → Mar harvest | Delta, irrigated areas |\n\n"
            "**Key Varieties:**\n"
            "- **ADT 37 / TKM 13** — Medium-duration (115–120 days); ideal Kharif; Coimbatore, Salem\n"
            "- **CO 51** — Short-duration (105 days); good for drought-stress conditions\n"
            "- **Samba Mahsuri Sub1** — Submergence-tolerant; flood-prone delta areas\n"
            "- **CR 1009 Sub1 (Swarna-Sub1)** — Tolerates 2–3 weeks submergence; Cuddalore, Cauvery delta\n"
            "- **ASD 16** — Salt-tolerant; coastal Chennai, Tirunelveli\n"
            "- **TRY 3** — Drought-tolerant; Erode, Salem uplands\n\n"
            "**Irrigation:** Paddy needs 1200–1500 mm; AWD technique saves 20–30% water. "
            "Apply 2 cm standing water only during tillering and panicle initiation."
        )
    ),

    # ── DROUGHT CROPS & MILLETS ───────────────────────────────────────────────
    (
        ["millet", "sorghum", "jowar", "bajra", "pearl millet", "thinai",
         "kudiraivali", "varagu", "kambu", "ragi", "finger millet", "kuthiraivali",
         "drought crop", "dryland crop", "dry land"],
        (
            "🌿 **Drought-Tolerant Millets & Dryland Crops for Tamil Nadu**\n\n"
            "When drought risk is **Moderate–High**, shift paddy area to these crops:\n\n"
            "| Crop | Tamil Name | Duration | Districts | Water Need |\n"
            "|------|-----------|----------|-----------|------------|\n"
            "| Pearl Millet | Kambu | 75–85 days | Coimbatore, Salem, Vellore | Very Low |\n"
            "| Finger Millet | Ragi/Kezhvaragu | 90–120 days | Erode, Coimbatore | Low |\n"
            "| Foxtail Millet | Thinai | 70–80 days | Salem, Erode uplands | Very Low |\n"
            "| Kodo Millet | Varagu | 90–100 days | Erode, Madurai | Low |\n"
            "| Barnyard Millet | Kuthiraivali | 65–75 days | All dryland zones | Very Low |\n"
            "| Sorghum | Cholam | 100–115 days | All semi-arid zones | Low |\n\n"
            "**Intercropping tip:** Kambu + Horsegram (2:1 ratio) boosts income by 30% in dry years. "
            "Mulch with paddy straw to conserve soil moisture."
        )
    ),

    # ── PULSES ───────────────────────────────────────────────────────────────
    (
        ["pulse", "blackgram", "black gram", "greengram", "green gram", "horsegram",
         "horse gram", "cowpea", "redgram", "red gram", "vamban", "lentil", "dal"],
        (
            "🫘 **Pulse Crops for Tamil Nadu**\n\n"
            "**Blackgram (Ulundu):**\n"
            "- Varieties: **Vamban-8** (65 days), CO 6 (65 days), VBN 10\n"
            "- Season: Rabi (Dec–Jan planting); excellent post-paddy crop\n"
            "- Districts: All; especially Vellore, Tirunelveli, Madurai\n\n"
            "**Greengram (Pachai Payiru):**\n"
            "- Varieties: CO 8, Vamban-7 (65 days)\n"
            "- Summer and Rabi; drought-tolerant\n\n"
            "**Horsegram (Kollu):**\n"
            "- Most drought-tolerant pulse; ideal for **very low rainfall zones**\n"
            "- Varieties: PAIYUR 1, CO 1; grown in Erode, Salem, Dharmapuri\n"
            "- Fixes ~70 kg N/ha — excellent green manure and soil restorer\n\n"
            "**Redgram / Pigeon Pea (Toor Dal):**\n"
            "- Duration 150–180 days; semi-perennial\n"
            "- Varieties: CO 6, ICPL 87; grown in Vellore, Tirunelveli\n\n"
            "**Note:** All pulses fix atmospheric nitrogen — include 1 pulse crop per year in rotation to reduce urea input by 20–25%."
        )
    ),

    # ── GROUNDNUT / OILSEEDS ─────────────────────────────────────────────────
    (
        ["groundnut", "peanut", "gingelly", "sesame", "sunflower", "oil seed",
         "oilseed", "coconut", "castor"],
        (
            "🥜 **Oilseed Crops for Tamil Nadu**\n\n"
            "**Groundnut (Verkadalai):**\n"
            "- Varieties: VRI 2, CO 7, TMV 13, GPBD 4 (resistant to late leaf spot)\n"
            "- Duration: 105–125 days; Kharif and Rabi\n"
            "- Districts: Vellore, Tirunelveli, Erode (sandy loam preferred)\n"
            "- Irrigation: 400–550 mm; critical at pegging and pod filling stages\n\n"
            "**Gingelly / Sesame (Ellu):**\n"
            "- Varieties: TMV 7, VRI 1; 75–85 days; very low water need\n"
            "- Ideal for light soils during summer and Rabi\n\n"
            "**Sunflower:**\n"
            "- Varieties: KBSH 44, CO 4; 90–100 days\n"
            "- Good Rabi crop in Coimbatore and Erode\n\n"
            "**Coconut (Thengai):**\n"
            "- Perennial; 50–100 liters/palm/day irrigation by drip saves 40%\n"
            "- Districts: Coimbatore, Tirunelveli, Cuddalore coast"
        )
    ),

    # ── SUGARCANE ─────────────────────────────────────────────────────────────
    (
        ["sugarcane", "sugar cane", "cane", "saccharum"],
        (
            "🎋 **Sugarcane in Tamil Nadu**\n\n"
            "- Major varieties: **Co 0238** (high sugar), Co 86032, CoSi 95071\n"
            "- Duration: 12–18 months (plant crop) + 2–3 ratoon cycles\n"
            "- Districts: Erode (largest), Salem, Cuddalore, Tirunelveli\n\n"
            "**Water Management:**\n"
            "- Needs 1500–2500 mm; drip irrigation saves 40–50% water\n"
            "- Critical periods: germination, tillering, grand growth phase\n"
            "- Use **trash mulching** (dry leaves) to reduce soil moisture loss 30%\n\n"
            "**During drought:** Reduce fertilizer, apply potassium to reduce transpiration, "
            "skip ratoon and replant next season.\n\n"
            "**During flood:** Ensure drainage within 24 hours; apply 2% urea foliar spray "
            "post-flooding to recover nutrient losses."
        )
    ),

    # ── COTTON ───────────────────────────────────────────────────────────────
    (
        ["cotton", "kapas", "bt cotton"],
        (
            "🌿 **Cotton Cultivation in Tamil Nadu**\n\n"
            "- Varieties: MCU 7, DCH 32 (hybrid), MECH 162 BG II (Bt)\n"
            "- Duration: 150–180 days (long season) or 120–150 days (medium)\n"
            "- Districts: Coimbatore, Madurai, Tirunelveli, Erode\n"
            "- Season: Kharif (June–July planting)\n\n"
            "**Key Practices:**\n"
            "- Black cotton soil (Vertisol) is ideal — good water retention\n"
            "- Drip irrigation + fertigation: 35–40% water savings\n"
            "- **Pink bollworm management:** Use pheromone traps (5/ha); spray Emamectin benzoate\n"
            "- Avoid planting when La Niña brings excessive NEM rains — boll rotting risk\n\n"
            "**During drought:** Apply 1% KNO₃ foliar spray; mulch inter-rows to conserve moisture."
        )
    ),

    # ── IRRIGATION & WATER ───────────────────────────────────────────────────
    (
        ["irrigation", "drip", "awd", "alternate wetting", "sprinkler", "water",
         "watering", "canal", "reservoir", "tank", "kulambu", "dam"],
        (
            "💧 **Irrigation & Water Management in Tamil Nadu**\n\n"
            "**Alternate Wetting & Drying (AWD) for Paddy:**\n"
            "- Let soil dry until water level drops 15 cm below surface, then irrigate to 3 cm\n"
            "- Saves 20–30% water with no yield loss\n"
            "- Most effective in Cauvery delta (Thanjavur, Trichy)\n\n"
            "**Drip Irrigation:**\n"
            "- Mandatory for: Sugarcane, Banana, Coconut, Groundnut, Cotton\n"
            "- Subsidy: 45–90% under PM-KUSUM / Tamil Nadu Horticultural Mission\n"
            "- Saves 40–50% water; 20–30% yield increase with fertigation\n\n"
            "**Sprinkler:**\n"
            "- Ideal for: Groundnut, Pulses, Millets, Vegetables\n"
            "- Best for undulating terrain (Erode, Salem, Vellore)\n\n"
            "**Flood/Deficit Response:**\n"
            "- Flood: Clear silt from field channels within 48 hrs; pump drainage\n"
            "- Drought: Prioritize critical crop growth stages; skip irrigation at vegetative stage\n\n"
            "**Reservoirs & Tanks:** Mettur Dam (Salem/Erode), Krishnagiri, Amaravathi — "
            "check Tamil Nadu PWD water levels before planning irrigation schedule."
        )
    ),

    # ── DROUGHT MANAGEMENT ───────────────────────────────────────────────────
    (
        ["drought", "dry", "water stress", "moisture deficit", "deficit",
         "low rainfall", "rain fail", "no rain", "rainfall deficit"],
        (
            "🏜️ **Drought Management Advisory for Tamil Nadu**\n\n"
            "**Immediate Actions (first 2 weeks of deficit):**\n"
            "1. Activate AWD (Alternate Wetting & Drying) for paddy fields\n"
            "2. Switch irrigation to early morning (4–7 AM) to minimize evaporation\n"
            "3. Apply **soil moisture conservation mulch** (paddy straw, sugarcane trash)\n"
            "4. Postpone transplanting / direct seeding — wait for the next rain forecast\n\n"
            "**Crop Choices Under Drought:**\n"
            "- Replace paddy with: **Pearl Millet (Kambu), Finger Millet (Ragi), Horsegram, Cowpea**\n"
            "- Use drought-tolerant paddy varieties if paddy must be grown: **CO 51, TRY 3, ADT 45**\n\n"
            "**Soil Management:**\n"
            "- Deep ploughing (30 cm) before monsoon to improve water infiltration\n"
            "- Contour bunding and farm ponds for water harvesting\n"
            "- Apply 25 kg/ha potassium (MOP) to improve plant drought tolerance\n\n"
            "**Government Schemes:** Pradhan Mantri Fasal Bima Yojana (PMFBY) — crop insurance; "
            "apply through nearest Agriculture office or e-crop portal."
        )
    ),

    # ── FLOOD MANAGEMENT ─────────────────────────────────────────────────────
    (
        ["flood", "waterlog", "inundation", "submerg", "excessive rain",
         "heavy rain", "cyclone", "cyclonic"],
        (
            "🌊 **Flood / Waterlogging Management Advisory for Tamil Nadu**\n\n"
            "**Emergency Field Actions (first 24–48 hours):**\n"
            "1. Open all drainage channels and bunds to drain water as quickly as possible\n"
            "2. Do NOT apply fertilizer immediately after flooding — wash-out loss\n"
            "3. Pump-out stagnant water from paddy fields within 48–72 hours (>72 hrs = severe loss)\n\n"
            "**Submergence-Tolerant Varieties (plan for next season):**\n"
            "- **Swarna-Sub1 / CR 1009 Sub1** — tolerates 14–18 days submergence\n"
            "- **Samba Mahsuri Sub1** — tolerates 12 days\n"
            "- Suitable for Cauvery Delta (Thanjavur, Cuddalore, Tiruchy)\n\n"
            "**Post-Flood Recovery:**\n"
            "- 48 hrs after drainage: Apply 2% urea foliar spray to restore leaf nitrogen\n"
            "- Spray potassium (0.5% K₂SO₄) to enhance recovery\n"
            "- Re-sow where total stand loss occurs (keep nursery rows as backup)\n\n"
            "**Banana & Horticultural Crops:** De-bunch early; apply Bordeaux mixture "
            "to stem cuts to prevent fungal entry. Replant after 2–3 weeks drying."
        )
    ),

    # ── SOWING WINDOWS ───────────────────────────────────────────────────────
    (
        ["sowing", "planting", "nursery", "transplant", "seed", "seedling",
         "kharif", "rabi", "sornavari", "pishanam"],
        (
            "🗓️ **Tamil Nadu Crop Sowing Calendar**\n\n"
            "| Season | Months | Main Crops |\n"
            "|--------|--------|------------|\n"
            "| **Sornavari / Early Kharif** | Apr–May | Groundnut, Vegetables, Maize |\n"
            "| **Kharif / Kuruvai (Paddy)** | Jun–Jul | Short-duration paddy (CO 51, ADT 37) |\n"
            "| **Samba (Main Paddy)** | Jul–Aug | Long/medium paddy (ADT 45, ASD 16) |\n"
            "| **Rabi / Thaladi** | Oct–Nov | Pulses, Ragi, Blackgram, Thaladi paddy |\n"
            "| **Pishanam / Winter** | Dec–Jan | Blackgram, Greengram, Potato (Nilgiris) |\n\n"
            "**Nursery Tips:**\n"
            "- Raise nursery in raised beds 30 days before main field transplanting\n"
            "- Treat seeds with 3 g Thiram / 1 g Carbendazim per kg seed before sowing\n"
            "- Use **staggered sowing** (two tranches 10 days apart) to buffer monsoon onset uncertainty\n\n"
            "**Coimbatore (Western Zone):** Main sowing Jun 15–Jul 15 (SWM onset)\n"
            "**Delta Districts (Thanjavur, Cuddalore):** Samba planting Aug 1–Sep 15 (reservoir release)\n"
            "**Tirunelveli / Madurai:** NEM-dependent — plant Rabi after Oct 15 NEM onset"
        )
    ),

    # ── SOIL HEALTH ──────────────────────────────────────────────────────────
    (
        ["soil", "soil health", "organic matter", "compost", "vermicompost",
         "green manure", "biofertilizer", "rhizobium", "azospirillum", "ph",
         "soil test", "vertisol", "alfisol", "red soil", "black soil"],
        (
            "🌍 **Soil Health Management for Tamil Nadu**\n\n"
            "**Tamil Nadu Soil Types:**\n"
            "| Zone | Soil Type | Key Issue |\n"
            "|------|-----------|----------|\n"
            "| Coimbatore, Erode | Red loamy (Alfisol) | Low water retention, slightly acidic |\n"
            "| Cauvery Delta | Heavy clay (Vertisol) | Waterlogging in monsoon |\n"
            "| Coastal zones | Sandy / saline | Salinity, low nutrients |\n"
            "| Nilgiris | Hill loam | Erosion, leaching |\n\n"
            "**Soil Testing:** Get tested every 3 years at district Soil Testing Lab (free under INM scheme)\n"
            "- Optimal pH: 6.0–7.5 for most crops; paddy tolerates 5.5–7.0\n"
            "- Below pH 5.5: Apply **2 t/ha agricultural lime** or **Dolomite**\n\n"
            "**Organic Amendments:**\n"
            "- FYM (Farm Yard Manure): 12.5 t/ha before main crop\n"
            "- Vermicompost: 2.5 t/ha; improves soil structure and water holding\n"
            "- Green manure (Dhaincha/Sunnhemp): Incorporate 45 days after sowing → adds 80 kg N/ha\n\n"
            "**Biofertilizers (free from Dept. of Agriculture TN):**\n"
            "- Rhizobium for pulses; Azospirillum for cereals; PSB for phosphorus mobilization\n"
            "- Use with reduced urea (save 25–30% N cost)"
        )
    ),

    # ── FERTILIZER ───────────────────────────────────────────────────────────
    (
        ["fertilizer", "npk", "urea", "dap", "nitrogen", "phosphorus",
         "potassium", "micronutrient", "zinc", "boron", "fertigation",
         "manure", "organic"],
        (
            "🧪 **Fertilizer & Nutrient Management**\n\n"
            "**Paddy Recommendation (per hectare):**\n"
            "- N: 120 kg (split into 3: basal 40%, tillering 30%, panicle initiation 30%)\n"
            "- P₂O₅: 50 kg (full basal application)\n"
            "- K₂O: 50 kg (basal 50% + panicle 50%)\n\n"
            "**Dryland Crops (Millets, Pulses):**\n"
            "- Reduce N by 25% if biofertilizers used\n"
            "- Apply K during stress periods to improve drought tolerance\n\n"
            "**Micronutrients:**\n"
            "- **Zinc deficiency** (khaira disease in paddy): Apply 25 kg ZnSO₄/ha or "
            "spray 0.5% ZnSO₄ at 2 weeks after transplanting\n"
            "- **Boron** for groundnut, sunflower: 1 kg/ha boranated superphosphate\n"
            "- **Iron chlorosis** in calcareous soils: spray 0.5% FeSO₄ + 0.1% citric acid\n\n"
            "**Top Tip:** Use **Soil Health Card** (free from Krishi Vigyan Kendra) "
            "to get district-specific NPK + micronutrient recommendations."
        )
    ),

    # ── PEST & DISEASE ───────────────────────────────────────────────────────
    (
        ["pest", "disease", "fungal", "blight", "blast", "sheath", "brownspot",
         "virus", "borer", "aphid", "whitefly", "thrips", "weed", "herbicide",
         "pesticide", "insecticide", "spray", "ipm", "biocontrol"],
        (
            "🐛 **Pest & Disease Management (IPM Approach)**\n\n"
            "**Paddy — Common Problems:**\n"
            "| Problem | Symptom | IPM Action |\n"
            "|---------|---------|------------|\n"
            "| Blast (Magnaporthe oryzae) | Brown diamond lesions on leaves | Spray Tricyclazole 75 WP 0.6 g/L |\n"
            "| Sheath blight | White-brown lesions on sheath | Hexaconazole 1 ml/L |\n"
            "| Brown Plant Hopper | Yellow hopperburn patches | Buprofezin 25 SC 1.6 ml/L; light trap |\n"
            "| Stem borer | Dead heart / white ear | Cartap Hydrochloride 4G 18.75 kg/ha |\n\n"
            "**Dryland Crops:**\n"
            "- Groundnut early leaf spot: Mancozeb 75 WP 2 g/L at 30-day intervals\n"
            "- Cotton bollworm: Pheromone traps (5/ha); Emamectin 5 SG 0.3 g/L\n"
            "- Pulse pod borer: Bt spray (Bacillus thuringiensis) + NSKE 5%\n\n"
            "**Weed Management:**\n"
            "- Direct seeded paddy: Pyrazosulfuron-ethyl (10 g/ha) at 8–10 DAS\n"
            "- Transplanted paddy: Hand weeding at 20 and 45 DAT\n\n"
            "**Biocontrol agents (free/subsidized from Dept. Agri TN):**\n"
            "- Trichogramma cards for stem borer; Pseudomonas for soil-borne diseases"
        )
    ),

    # ── CROP ROTATION ────────────────────────────────────────────────────────
    (
        ["rotation", "crop sequence", "intercrop", "intercropping", "mixed crop",
         "fallow", "crop calendar"],
        (
            "🔄 **Crop Rotation & Intercropping for Tamil Nadu**\n\n"
            "**Cauvery Delta Zone (Thanjavur, Cuddalore):**\n"
            "- Kuruvai paddy → Samba paddy → Blackgram (Rabi)\n"
            "- Or: Paddy → Paddy → Sunflower (avoids blast build-up)\n\n"
            "**Western Zone (Coimbatore, Erode):**\n"
            "- Groundnut → Maize → Blackgram\n"
            "- Or: Sugarcane (1yr) → Ratoon → Pulses\n\n"
            "**Southern Zone (Madurai, Tirunelveli):**\n"
            "- Samba paddy → Gingelly → Cowpea\n"
            "- Or: Cotton (Kharif) → Blackgram (Rabi)\n\n"
            "**Key Intercropping Combos:**\n"
            "- Maize + Cowpea (2:1 rows) — income insurance\n"
            "- Sugarcane + Blackgram (trash mulch rows) — 20% extra income\n"
            "- Groundnut + Castor (8:2 rows) — pest diversification\n\n"
            "**Rule:** Include at least **one pulse or legume per year** to naturally fix soil nitrogen."
        )
    ),

    # ── DISTRICT-SPECIFIC ZONES ──────────────────────────────────────────────
    (
        ["thanjavur", "cauvery delta", "delta zone"],
        (
            "📍 **Thanjavur — Cauvery Delta Zone**\n\n"
            "- **Soil:** Heavy clay (Vertisol), high fertility, pH 7–8\n"
            "- **Main crops:** Paddy (Samba, Kuruvai, Thaladi), Banana, Sugarcane\n"
            "- **Irrigation:** Cauvery system + Grand Anicut canal\n"
            "- **Key risk:** Cyclone-driven NEM flooding (Oct–Nov); waterlogging\n"
            "- **Advisory:** Use Swarna-Sub1 paddy; clear drainage bunds before NEM; "
            "avoid Samba planting after Sep 15 to prevent cold injury at panicle stage"
        )
    ),
    (
        ["coimbatore", "western zone", "western agro"],
        (
            "📍 **Coimbatore — Western Agro-Climatic Zone**\n\n"
            "- **Soil:** Red loamy (Alfisol), well-drained, pH 6.0–7.5\n"
            "- **Main crops:** Sugarcane (largest area), Cotton, Maize, Groundnut, Coconut\n"
            "- **Irrigation:** Bhavani river, Amaravathi reservoir, bore wells (falling water table!)\n"
            "- **Key risk:** El Niño-driven SWM deficit; summer moisture stress\n"
            "- **Advisory:** Drip irrigation mandatory for sugarcane; "
            "shift to ADT 37 paddy if Bhavani levels drop; drill millets in rain-shadow months"
        )
    ),
    (
        ["madurai", "southern zone"],
        (
            "📍 **Madurai — Southern Agro-Climatic Zone**\n\n"
            "- **Soil:** Black cotton soil (Vertisol), pH 7.5–8.5\n"
            "- **Main crops:** Cotton, Paddy (Samba), Banana, Jasmine (floriculture)\n"
            "- **Irrigation:** Vaigai river, Periyar (Mullai Periyar) reservoir\n"
            "- **Key risk:** Inter-seasonal drought; high temperature stress (Apr–Jun)\n"
            "- **Advisory:** Plant cotton after first SWM rains (Jun 15+); "
            "use heat-tolerant paddy CO 51; drip-irrigate banana"
        )
    ),
    (
        ["chennai", "coastal zone", "northern coastal"],
        (
            "📍 **Chennai — Northern Coastal Zone**\n\n"
            "- **Soil:** Sandy loam, saline patches near coast, pH 7–8.5\n"
            "- **Main crops:** Paddy (ASD 16 salt-tolerant), Vegetables, Coconut\n"
            "- **Irrigation:** Poondi, Chembarambakkam reservoirs; bore wells (salinity risk)\n"
            "- **Key risk:** Cyclones (Oct–Dec), urban heat island, saline intrusion\n"
            "- **Advisory:** Grow ASD 16 or CST 7-1 salt-tolerant varieties; "
            "avoid irrigating with saline bore water (EC > 2 dS/m) during sensitive stages"
        )
    ),
    (
        ["vellore", "north western zone"],
        (
            "📍 **Vellore — North Western Zone**\n\n"
            "- **Soil:** Red soil (Alfisol), pH 5.5–6.5\n"
            "- **Main crops:** Groundnut, Paddy, Sorghum, Banana\n"
            "- **Irrigation:** Palar river (seasonal), check dams, drip systems\n"
            "- **Key risk:** Low and erratic rainfall (600–750 mm/yr), drought-prone\n"
            "- **Advisory:** Prioritize groundnut (VRI 2, GPBD 4) and horsegram; "
            "construct farm ponds under MNREGA scheme; spray TNAU's drought-mitigation package at heading"
        )
    ),

    # ── GOVERNMENT SCHEMES ───────────────────────────────────────────────────
    (
        ["scheme", "subsidy", "government", "pmfby", "pm kisan", "kisan", "tnau",
         "kvk", "krishi vigyan", "mgnrega", "mnrega", "atma", "insurance",
         "compensation", "relief"],
        (
            "🏛️ **Tamil Nadu Agriculture Government Schemes**\n\n"
            "| Scheme | Benefit | How to Apply |\n"
            "|--------|---------|-------------|\n"
            "| **PM-KISAN** | ₹6,000/yr income support | PM Kisan portal / CSC |\n"
            "| **PMFBY** | Crop insurance (drought/flood/pest) | Bank / Aadhaar + land records |\n"
            "| **TN Drip Subsidy** | 45–90% cost for drip/sprinkler | Block Agri Office |\n"
            "| **Soil Health Card** | Free soil testing every 3 yrs | District lab |\n"
            "| **TNAU Seed Subsidy** | Certified seeds at 50% discount | Agri. cooperative society |\n"
            "| **MNREGA Ponds** | Free farm pond construction (100 m²) | Gram Panchayat |\n"
            "| **KVK (Krishi Vigyan Kendra)** | Free training, demonstrations | Visit nearest KVK |\n\n"
            "**Emergency Relief:** Contact District Collector / Agriculture Dept. within 72 hrs of crop loss "
            "for damage assessment and ex-gratia under SDRF/NDRF."
        )
    ),

    # ── CLIMATE CHANGE ───────────────────────────────────────────────────────
    (
        ["climate change", "global warming", "temperature rise", "carbon",
         "greenhouse", "adaptation", "resilience", "climate resilient"],
        (
            "🌡️ **Climate Change Adaptation for Tamil Nadu Farmers**\n\n"
            "**Observed Trends in TN:**\n"
            "- Average temperature rising ~0.3–0.5°C per decade\n"
            "- More intense but less frequent rainfall events\n"
            "- NEM onset becoming less predictable (±2–3 weeks variability)\n\n"
            "**Adaptation Strategies:**\n"
            "1. **Variety switch:** Use shorter-duration varieties to escape heat stress at flowering\n"
            "2. **Sowing date adjustment:** Shift sowing 2–3 weeks earlier to avoid hot spell at harvest\n"
            "3. **Agroforestry:** Intercrop trees (Moringa, Neem, Silver Oak) for shade and microclimate cooling\n"
            "4. **Water harvesting:** Farm ponds, percolation tanks, contour bunding\n"
            "5. **SRI Method** (System of Rice Intensification): 25% more yield, 30% less water\n"
            "6. **Solar pumps:** PM-KUSUM subsidy (90%) — replace diesel pumps; "
            "reduces cost and carbon footprint\n\n"
            "**TNAU Climate-Smart Agriculture Programme:** Free village-level advisory through "
            "Agri Business Development Centres (ABDCs)"
        )
    ),

    # ── HORTICULTURE / VEGETABLES / FRUIT ───────────────────────────────────
    (
        ["vegetable", "tomato", "brinjal", "onion", "chilli", "banana", "mango",
         "papaya", "guava", "moringa", "drumstick", "horticulture", "fruit"],
        (
            "🍅 **Horticulture & Vegetable Crops in Tamil Nadu**\n\n"
            "**Key Vegetable Crops:**\n"
            "| Crop | Season | Districts | Key Variety |\n"
            "|------|--------|-----------|-------------|\n"
            "| Tomato | Oct–Feb | Coimbatore, Salem | CO 3, TNAU Hybrid-1 |\n"
            "| Brinjal | Year-round | All | CO 1, PLR 1 |\n"
            "| Chilli | Jun–Aug | Madurai, Coimbatore | K1, CO 4 |\n"
            "| Onion | Nov–Feb | Erode, Salem | Arka Niketan |\n\n"
            "**Fruit Crops:**\n"
            "- **Banana (Cavendish, Rasthali, Poovan):** Trichy, Erode; 11–12 months; "
            "drip + fertigation essential; tissue culture plants increase yield 30%\n"
            "- **Mango (Alphonso, Bangalora, Neelum):** Salem, Dharmapuri; "
            "Paclobutrazol spray in Nov for off-season flowering\n"
            "- **Moringa/Drumstick (PKM 1):** High-value, drought-tolerant; "
            "Tirunelveli specialty — 8 t/ha pods/yr with minimal water\n\n"
            "**Protected Cultivation:** Poly-houses for tomato, capsicum, rose — "
            "65% subsidy under National Horticulture Mission (NHM)"
        )
    ),

    # ── GENERAL / FALLBACK ───────────────────────────────────────────────────
    (
        ["help", "what can you", "what do you", "what is your", "tell me",
         "explain", "how to", "how do i", "which crop", "which variety"],
        (
            "🤔 I'm here to help with any Tamil Nadu agriculture question!\n\n"
            "Try asking:\n"
            "- *'Which crops are best during El Niño?'*\n"
            "- *'What is AWD irrigation?'*\n"
            "- *'Best paddy variety for flood-prone area?'*\n"
            "- *'How to manage soil in Coimbatore?'*\n"
            "- *'What government schemes are available?'*\n"
            "- *'Sowing dates for blackgram in Thanjavur?'*"
        )
    ),
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, condense whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_entry(query_norm: str, patterns: list) -> int:
    """Return number of keyword patterns matched in query."""
    return sum(1 for kw in patterns if kw in query_norm)


def get_agriculture_response(
    user_query: str,
    selected_district: Optional[str] = None,
    advisory: Optional[dict] = None
) -> str:
    """
    Returns a rule-based agriculture advisory response for a farmer query.

    Parameters
    ----------
    user_query       : Raw text query from the user
    selected_district: Currently selected district (for context injection)
    advisory         : Current advisory dict from src/advisory.py (for context injection)

    Returns
    -------
    str: Markdown-formatted advisory response
    """
    q = _normalize(user_query)

    # Score all knowledge base entries
    scored = []
    for patterns, response in AGRI_KB:
        score = _score_entry(q, patterns)
        if score > 0:
            scored.append((score, response))

    # Sort by match score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        best_response = scored[0][1]

        # Inject district context into the response if no district keyword was in query
        if selected_district and selected_district.lower() not in q:
            best_response = (
                f"*📍 Responding for **{selected_district}** context:*\n\n"
                + best_response
            )
        return best_response

    # Fallback: try to be helpful based on advisory context
    if advisory:
        zone = advisory.get("agro_climatic_zone", "Tamil Nadu")
        crops = ", ".join(
            [c.get("crop", "") for c in advisory.get("recommended_crops", [])[:3]]
        )
        return (
            f"🌾 Based on the current forecast for **{selected_district or 'your district'}** "
            f"({zone}), the top recommended crops are: **{crops}**.\n\n"
            f"You can ask me more specific questions about:\n"
            f"- ENSO / IOD effects on rainfall\n"
            f"- Drought or flood management\n"
            f"- Sowing windows, fertilizers, pests\n"
            f"- Government schemes and subsidies"
        )

    return (
        "🙏 I didn't quite catch that. Try asking:\n"
        "- *'What is El Niño?'* or *'How does IOD affect Tamil Nadu?'*\n"
        "- *'Best crops for drought conditions?'*\n"
        "- *'Sowing window for paddy in Thanjavur?'*\n"
        "- *'How to manage flood in Cauvery delta?'*"
    )
