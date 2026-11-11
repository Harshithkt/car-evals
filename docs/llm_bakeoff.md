# Choosing the vision LLM for unknown cars

The trained models only know the cars in their class lists. When the all-cars model's confidence is low (usually a car outside
its list, such as an India-only model), the app asks a vision LLM on Nebius Token Factory instead. This page records how that
LLM was chosen (9 November 2026).

## Which models can see images

All 30 chat models on the API key were sent the same car photo. 16 rejected it ("This model does not support image input").
14 accepted it. The 8 that answered correctly and in reasonable time went to the test below; Nemotron-3.5-Super-VL (13.6 s,
wrong), Qwen3.8-27B and GLM-5.3-Flash (wrong), DeepSeek-V4.1-Flash (16.7 s), Kimi-K3 (97 s) and Kimi-K2.7-Code were left out.

## Test 1: 30 Indian-market cars

Photos from Wikimedia Commons, one per model: Kia Sonet, Tata Nexon, Mahindra Bolero, Tata Nano, Renault Kwid, Tata Tiago, Volkswagen Virtus, Maruti Suzuki Swift, Maruti Suzuki Baleno, Maruti Suzuki Dzire, Maruti Suzuki Brezza, Maruti Suzuki Ertiga, Maruti Suzuki Wagon R, Hyundai Creta, Hyundai Grand i10, Hyundai i20, Hyundai Verna, Tata Punch, Tata Altroz, Mahindra Thar, Mahindra XUV700, Mahindra Scorpio, Toyota Innova, Toyota Fortuner, Honda City, Honda Amaze, Kia Carens, Skoda Kushaq, Renault Duster, Nissan Magnite. Three photos were dropped before the test because they were unfair (a
brake close-up, a concept car, a tiny car among many). Each model got the exact prompt the app uses
([`backend/llm.py`](../backend/llm.py)), with up to 1,500 output tokens so reasoning models can finish. An answer counts as the
right model when the make and model match, allowing regional names of the same car (Creta = ix25, Verna = Accent/Solaris,
Innova Hycross = Zenix, Wagon R = Karimun, Renault Duster = Dacia Duster).

| Model | Exact model | Make | Valid JSON | Median time | 90th percentile | Tokens per call |
|---|---|---|---|---|---|---|
| `meta-models/muse-spark-open` | 30/30 | 30/30 | 30/30 | 3.6 s | 4.6 s | 1322 |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | 26/30 | 27/30 | 28/30 | 5.9 s | 10.0 s | 908 |
| `MiniMaxAI/MiniMax-M3` | 24/30 | 28/30 | 30/30 | 3.2 s | 4.0 s | 1082 |
| `google/gemma-3-27b-it` | 22/30 | 25/30 | 30/30 | 2.8 s | 4.1 s | 484 |
| `moonshotai/Kimi-K3.1-Pro` | 21/30 | 21/30 | 22/30 | 4.0 s | 5.8 s | 1490 |
| `openbmb/MiniCPM-V-4_5` | 20/30 | 27/30 | 30/30 | 1.9 s | 2.1 s | 473 |
| `MiniMaxAI/MiniMax-M3.1-Flash` | 18/30 | 22/30 | 28/30 | 4.2 s | 6.8 s | 1173 |
| `moonshotai/Kimi-K2.6` | 17/30 | 17/30 | 18/30 | 5.2 s | 6.4 s | 1673 |

## Test 2: cars the trained models know

The 12 recent-car samples of the all-cars model and the 6 Stanford Cars sample images, cropped like in the app:

| Model | Exact model | Make |
|---|---|---|
| `meta-models/muse-spark-open` | 18/18 | 18/18 |
| `google/gemma-3-27b-it` | 13/18 | 18/18 |

muse-spark-open often named the exact generation code too (Nissan Leaf ZE2, Range Rover Evoque L551, Lexus GX J250, Chevrolet
Silverado GMT900).

## Decision

`meta-models/muse-spark-open` is the default (48/48 across both tests, about 3.6 s per call). If a key cannot use it, the
backend falls back to `google/gemma-3-27b-it`. `NEBIUS_VLM_MODEL` overrides the choice. This is a small test (48 photos), so
the LLM's answers are still shown as unevaluated in the app and never count towards any metric.

## Photo credits (test 1)

- Kia Sonet: [2021 Kia Sonet 1.5 EX.jpg](https://commons.wikimedia.org/wiki/File:2021_Kia_Sonet_1.5_EX.jpg), CC BY-SA 2.0
- Tata Nexon: [2018 Tata Nexon XM.jpg](https://commons.wikimedia.org/wiki/File:2018_Tata_Nexon_XM.jpg), CC BY-SA 4.0
- Mahindra Bolero: [2009 Mahindra Bolero DC CRDe (7141667401).jpg](https://commons.wikimedia.org/wiki/File:2009_Mahindra_Bolero_DC_CRDe_(7141667401).jpg), CC BY 2.0
- Tata Nano: [2011 Tata Nano.jpg](https://commons.wikimedia.org/wiki/File:2011_Tata_Nano.jpg), CC BY 2.0
- Renault Kwid: [04-12-2023 Visita a Fábrica da Renault em São José dos Pinhais PR (53375902839).jpg](https://commons.wikimedia.org/wiki/File:04-12-2023_Visita_a_F%C3%A1brica_da_Renault_em_S%C3%A3o_Jos%C3%A9_dos_Pinhais_PR_(53375902839).jpg), CC BY 2.0
- Tata Tiago: [2016-03-01 Geneva Motor Show G030.JPG](https://commons.wikimedia.org/wiki/File:2016-03-01_Geneva_Motor_Show_G030.JPG), CC BY-SA 4.0
- Volkswagen Virtus: [2018 Volkswagen Virtus 1.6 MSi Highline AT.jpg](https://commons.wikimedia.org/wiki/File:2018_Volkswagen_Virtus_1.6_MSi_Highline_AT.jpg), CC BY 4.0
- Maruti Suzuki Swift: [Maruti Suzuki Swift LXi.jpg](https://commons.wikimedia.org/wiki/File:Maruti_Suzuki_Swift_LXi.jpg), CC BY 2.0
- Maruti Suzuki Baleno: [2022 Maruti Suzuki Baleno Alpha (India) front view.jpg](https://commons.wikimedia.org/wiki/File:2022_Maruti_Suzuki_Baleno_Alpha_(India)_front_view.jpg), CC BY 3.0
- Maruti Suzuki Dzire: [Maruti Suzuki Dzire VXi VVT - Subcompact Car - Kolkata 2018-01-17 7574.JPG](https://commons.wikimedia.org/wiki/File:Maruti_Suzuki_Dzire_VXi_VVT_-_Subcompact_Car_-_Kolkata_2018-01-17_7574.JPG), CC BY 3.0
- Maruti Suzuki Brezza: [2021 Maruti Suzuki Vitara Brezza VXI.jpg](https://commons.wikimedia.org/wiki/File:2021_Maruti_Suzuki_Vitara_Brezza_VXI.jpg), CC BY-SA 4.0
- Maruti Suzuki Ertiga: [Maruti Suzuki Ertiga(2).jpg](https://commons.wikimedia.org/wiki/File:Maruti_Suzuki_Ertiga(2).jpg), CC0
- Maruti Suzuki Wagon R: [Suzuki Karimun Wagon R (Front), Jakarta, Indonesia.jpg](https://commons.wikimedia.org/wiki/File:Suzuki_Karimun_Wagon_R_(Front),_Jakarta,_Indonesia.jpg), CC0
- Hyundai Creta: [HYUNDAI CRETA , iX25 (SU2) China (1).jpg](https://commons.wikimedia.org/wiki/File:HYUNDAI_CRETA_,_iX25_(SU2)_China_(1).jpg), CC BY-SA 4.0
- Hyundai Grand i10: [Hyundai Grand I10 Sedan.jpg](https://commons.wikimedia.org/wiki/File:Hyundai_Grand_I10_Sedan.jpg), CC BY-SA 4.0
- Hyundai i20: [Hyundai i20 Coupe WRC, IAA 2017, Frankfurt (1Y7A3189).jpg](https://commons.wikimedia.org/wiki/File:Hyundai_i20_Coupe_WRC,_IAA_2017,_Frankfurt_(1Y7A3189).jpg), CC BY-SA 4.0
- Hyundai Verna: [HYUNDAI VERNA (HYUNDAI ACCENT) (RB,RC) China (9).jpg](https://commons.wikimedia.org/wiki/File:HYUNDAI_VERNA_(HYUNDAI_ACCENT)_(RB,RC)_China_(9).jpg), CC BY-SA 4.0
- Tata Punch: [Tata Punch Camo spotted at Visakhapatnam.jpg](https://commons.wikimedia.org/wiki/File:Tata_Punch_Camo_spotted_at_Visakhapatnam.jpg), CC BY-SA 4.0
- Tata Altroz: [Tata Altroz, GIMS 2019, Le Grand-Saconnex (GIMS0646).jpg](https://commons.wikimedia.org/wiki/File:Tata_Altroz,_GIMS_2019,_Le_Grand-Saconnex_(GIMS0646).jpg), CC BY-SA 4.0
- Mahindra Thar: [Mahindra Thar Photoshoot at Perupalem Beach (West Godavari District, AP, India) Djdavid.jpg](https://commons.wikimedia.org/wiki/File:Mahindra_Thar_Photoshoot_at_Perupalem_Beach_(West_Godavari_District,_AP,_India)_Djdavid.jpg), CC BY-SA 4.0
- Mahindra XUV700: [A black Mahindra XUV700 SUV in Ashiana Brahmananda, Jamshedpur, India (Ank Kumar, Infosys Limited) 04.jpg](https://commons.wikimedia.org/wiki/File:A_black_Mahindra_XUV700_SUV_in_Ashiana_Brahmananda,_Jamshedpur,_India_(Ank_Kumar,_Infosys_Limited)_04.jpg), CC BY-SA 4.0
- Mahindra Scorpio: [Mahindra Scorpio.jpg](https://commons.wikimedia.org/wiki/File:Mahindra_Scorpio.jpg), CC BY 2.0
- Toyota Innova: [Toyota (Innova) Zenix 2.0 V 2023 (1).jpg](https://commons.wikimedia.org/wiki/File:Toyota_(Innova)_Zenix_2.0_V_2023_(1).jpg), CC BY-SA 4.0
- Toyota Fortuner: [Toyota Fortuner, Cape Town (P1060077).jpg](https://commons.wikimedia.org/wiki/File:Toyota_Fortuner,_Cape_Town_(P1060077).jpg), CC BY-SA 4.0
- Honda City: [Honda City 2020.jpg](https://commons.wikimedia.org/wiki/File:Honda_City_2020.jpg), CC BY-SA 4.0
- Honda Amaze: [Honda Amaze front view.jpg](https://commons.wikimedia.org/wiki/File:Honda_Amaze_front_view.jpg), CC BY-SA 4.0
- Kia Carens: [Kia Carens 2024 Model.jpg](https://commons.wikimedia.org/wiki/File:Kia_Carens_2024_Model.jpg), CC BY-SA 4.0
- Skoda Kushaq: [Skoda Kushaq Front.jpg](https://commons.wikimedia.org/wiki/File:Skoda_Kushaq_Front.jpg), CC BY-SA 4.0
- Renault Duster: [Dacia Duster III IMG 8970.jpg](https://commons.wikimedia.org/wiki/File:Dacia_Duster_III_IMG_8970.jpg), CC BY-SA 4.0
- Nissan Magnite: [Nissan Magnite 01.jpg](https://commons.wikimedia.org/wiki/File:Nissan_Magnite_01.jpg), CC BY-SA 4.0
