<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.md">English</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/dataset-desk/readme.png" alt="dataset-desk" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk"><img src="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/dataset-desk/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

<p align="center">
  A desk for making and reviewing visual datasets: find the copies, the near-copies and the odd ones out,<br>
  search a set by words or by picture, and check that every image's description really describes it.
</p>

---

## क्यों

एक स्टाइल LoRA अपनी प्रशिक्षण छवियों और उनके कैप्शन से सीखता है। दो मामूली दोष दोनों को खराब कर देते हैं:
- **दोहराव।** एक ही तस्वीर को दो बार सहेजा गया, या दो रेंडर कुछ पिक्सेल दूर, एक छवि को दो बार सिखाते हैं।
- **ऐसे विवरण जो विवरण नहीं हैं।** पीढ़ी प्रॉम्प्ट से कॉपी किया गया एक कैप्शन, या 54 छवियों पर चिपकाया गया एक नियम, मॉडल को तस्वीर के बारे में कुछ भी नहीं बताता है। इससे भी बदतर, यदि उस पाठ का उपयोग बाद में खोज का परीक्षण करने के लिए किया जाता है, तो यह प्रॉम्प्ट की रिकॉल को मापता है, न कि छवि की समझ को।

dataset-desk दोनों को एक प्रोजेक्ट फ़ोल्डर पर ढूंढता है, बिना इसे बदले। यह एक रिपोर्ट और एक समीक्षा कतार लिखता है ताकि कोई व्यक्ति उस पर कार्रवाई कर सके। **कुछ भी स्वचालित रूप से हटाया नहीं जाता है।**

## यह क्या करता है

| कमांड | आपको क्या मिलता है |
|---|---|
| `index` | प्रोजेक्ट को स्कैन करता है (रिकॉर्ड, वेव मेटाडेटा, PNG पीढ़ी डेटा) और प्रत्येक अद्वितीय छवि को `--out` के तहत एक कैश में एम्बेड करता है। |
| `dupes` | बाइट-समान प्रतियां (sha256), लगभग-डुप्लिकेट (एक दर्पण-जागरूक 256-बिट dHash) और सिमेंटिक डुप्लिकेट (एम्बेडिंग कोसाइन), JSON के रूप में। DINOv2 इंडेक्स पर, यह एक कैलिब्रेटेड दो-सिग्नल नियम का उपयोग करता है। एक ही वस्तु के दो दृश्य कभी भी जोड़े नहीं जाते हैं। |
| `search` | किसी विवरण (`--text`), एक तस्वीर (`--image`), या दोनों के सबसे निकट की छवियां |
| `coverage` | वांछित विषयों की सूची में प्रत्येक विषय के लिए सर्वोत्तम छवियां, ताकि अंतराल दिखाई दें |
| `report` | एक मार्कडाउन रिपोर्ट: डुप्लिकेट, सामग्री और रूप के अनुसार सबसे कम विशिष्ट स्वीकृत छवियां, विवरण ऑडिट, और क्या प्रत्येक विवरण अपनी छवि ढूंढता है |
| `text-index` | एक `--source` फ़ोल्डर (या एक rnd-data चेकआउट) की पाठ और कोड फ़ाइलों को एम्बेड करता है, मॉडल के अपने टोकनाइज़र के साथ चंक्स की गणना करता है |
| `text-dupes` | सटीक और लगभग-डुप्लिकेट फ़ाइलें। `--task code` केवल स्क्रिप्ट की तुलना करता है। |
| `text-search` | किसी क्वेरी (`--text`) के निकटतम फ़ाइलें। `--task code` मॉडल के कोड-पुनर्प्राप्ति प्रॉम्प्ट का उपयोग करता है। |
| `lineage` | रन में एक ही स्क्रिप्ट के संस्करण, समानता के अनुसार क्रमबद्ध |

**सीमाएं** स्टूडियो के अपने डेटा पर कैलिब्रेशन से आती हैं। प्रत्येक को कोड में अस्थायी के रूप में चिह्नित किया गया है, साथ ही उस माप के साथ जिससे यह आया है:
- **छवियां:** DINOv2 इंडेक्स पर, एक जोड़ी डुप्लिकेट होती है जब `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`। अन्य इंडेक्स 26 बिट्स के भीतर dHash का उपयोग करते हैं, साथ ही कोसाइन ≥ 0.95।
- **पाठ:** bge-m3 के साथ 0.98, और EmbeddingGemma 2 के साथ कोड के लिए 0.95। किसी अन्य मॉडल को `--semantic` की आवश्यकता होती है।

विभिन्न कैमरा कोणों से एक ही वस्तु के दो रेंडर कभी भी डुप्लिकेट नहीं होते हैं। दृश्य प्रत्येक रिकॉर्ड के `provenance.asset_id` और `render_id` से आता है, या `<asset>__y<yaw>_e<elev>` जैसे फ़ाइल नामों से।

**विवरण गेट** किसी भी मॉडल के बिना चलता है। जब कोई विवरण:
- अपने प्रॉम्प्ट को दोहराता है: यह प्रॉम्प्ट के बराबर है या इसके अंदर बैठता है, इसके साथ 6+ शब्दों का एक क्रम साझा करता है, या इसके 90% + शब्द (इसके 80% + सामग्री शब्द) इसमें दिखाई देते हैं;
- रिकॉर्ड की विषय पंक्ति को लपेटता है: 4+ शब्दों का `canon.subject` इसके शब्दों के एक सतत क्रम के रूप में दिखाई देता है;
- बहुत छोटा है: 8 शब्दों से कम या 5 सामग्री शब्दों से कम;
- सेट में किसी अन्य विवरण को दोहराता है: विराम चिह्न हटाने के बाद समान, या 0.85 या अधिक का सामग्री-शब्द जैकार्ड;
- SigLIP2 के लिए बहुत लंबा है: 64 टोकन से अधिक (एकमात्र जांच जिसके लिए एक टोकनाइज़र की आवश्यकता होती है; जांचे गए रिकॉर्ड को कभी भी पास के रूप में चिह्नित नहीं किया जाता है)।

चिह्नित रिकॉर्ड `description-review.jsonl` और एक **अंधा वर्कशीट** पर जाते हैं: प्रत्येक चिह्नित छवि के बगल में 3-5 दृश्य रूप से निकटतम भाई-बहन, बिना किसी प्रॉम्प्ट, विषय पंक्ति और दृश्य में पुराने पाठ के, ताकि इसे फिर से लिखने वाला व्यक्ति तस्वीर का वर्णन करे। एक वैकल्पिक ड्राफ्टर ([प्लेन-साइट](https://github.com/mcp-tool-shop-org/plain-sight), फ्लोरेंस-2) केवल पिक्सेल से एक पहला मसौदा लिख सकता है; मसौदे `captioner_assisted` के रूप में रिकॉर्ड किए जाते हैं, और एक व्यक्ति उन्हें संपादित करता है।

**लुक चैनल** प्रति छवि 30 संख्याएं हैं (मूल रिज़ॉल्यूशन पर पैलेट, टोन और बनावट), इसलिए "इस एक का रेंडरिंग गलत है" "इस एक का विषय अजीब है" से अलग दिखाई देता है।

## स्थापित करें

```bash
pip install dataset-desk
```

यह **लाइट टियर** देता है: केवल numpy और Pillow, कोई torch नहीं। डुप्लिकेट जांच, लुक चैनल, विवरण गेट और रिपोर्ट सभी इसमें काम करते हैं (`--backend none`)।

एम्बेडिंग इंडेक्स (खोज, कवरेज, सिमेंटिक डुप्लिकेट, विवरण फिट जांच) के लिए, मॉडल लाइब्रेरी जोड़ें:

```bash
pip install "dataset-desk[models]"
```

मॉडल पहली बार उपयोग किए जाने पर Hugging Face से पिन किए गए संशोधन पर प्राप्त किए जाते हैं, फिर आपके कैश से लोड होते हैं। प्रत्येक कार्य का एक डिफ़ॉल्ट होता है, और `--backend` दूसरा चुनता है:

| बैकएंड | मॉडल | डिफ़ॉल्ट के लिए |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | पाठ |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | कोड (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

EmbeddingGemma 2 का उपयोग उसी तरह किया जाता है जैसे कि इसके मॉडल कार्ड में कहा गया है: प्रत्येक कमांड के लिए एक कार्य उपसर्ग, पाठ के लिए केवल-पाठ लोडिंग, और `--dim 768|512|256`। `HF_HUB_OFFLINE=1` को डाउनलोड को रोकने के लिए सेट करें। प्लेन-साइट ड्राफ्टर हमेशा ऑफ़लाइन होता है: यह कभी भी डाउनलोड नहीं करता है।

**चलता है:** Python 3.11 और 3.12 पर, Windows और Linux पर। डिफ़ॉल्ट रूप से CPU, प्रत्येक थ्रेड पूल सीमित (4 थ्रेड जब तक कि `--threads` अन्यथा न कहे), और एक मॉडल रन प्रक्रिया को सामान्य से कम प्राथमिकता पर गिरा देता है, ताकि यह एक मशीन साझा कर सके।

## उपयोग

```bash
# The lite pass: no model, no GPU
dataset-desk index  --project path/to/project --backend none --out desk-out/project
dataset-desk report --project path/to/project --backend none --out desk-out/project

# With an embedding index
dataset-desk index    --project path/to/project --backend siglip2 --out desk-out/project
dataset-desk search   --project path/to/project --backend siglip2 --out desk-out/project --text "a squat bog creature"
dataset-desk coverage --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt
dataset-desk report   --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt

# Image duplicates with the calibrated DINOv2 rule
dataset-desk index --project path/to/project --backend dinov2 --out desk-out/project
dataset-desk dupes --project path/to/project --out desk-out/project

# Text and code files
dataset-desk text-index  --source path/to/folder --out desk-out/texts               # bge-m3, for text
dataset-desk text-dupes  --source path/to/folder --out desk-out/texts
dataset-desk text-index  --source path/to/folder --out desk-out/texts --task code   # EmbeddingGemma 2, for code
dataset-desk text-search --source path/to/folder --out desk-out/texts --task code --text "where the cache is written"
```

`dataset-desk --help` प्रत्येक ध्वज को सूचीबद्ध करता है। निकास कोड: `0` ठीक है, `1` एक उपयोगकर्ता त्रुटि या इनकार, `2` एक अप्रत्याशित त्रुटि। त्रुटियां `error <CODE>: <message>` के रूप में `hint:` पंक्ति के साथ मुद्रित होती हैं; `--debug` स्टैक ट्रेस जोड़ता है।

**यह जो प्रोजेक्ट लेआउट पढ़ता है** वह [स्टाइल-डेटासेट-लैब](https://github.com/mcp-tool-shop-org/style-dataset-lab) है:

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

एक ऐसा प्रोजेक्ट जिसमें कोई `records/` नहीं है, वह भी काम करता है, जब तक कि छवियां `outputs/` के तहत फ़ोल्डरों में हों (`outputs/` के शीर्ष पर फ़ाइलों को संपर्क शीट के रूप में छोड़ दिया जाता है): प्रत्येक छवि "अभिलेखित" है, और डुप्लिकेट और लुक जांच अभी भी चलती हैं।

## खतरा मॉडल और विश्वास

- **यह क्या पढ़ता है:** वह प्रोजेक्ट फ़ोल्डर जिसे आप `--project` नाम से बनाते हैं: छवियां, `records/*.json`, वेव मेटाडेटा और पीएनजी फ़ाइलों में एम्बेडेड जनरेशन डेटा। टेक्स्ट कमांड `--source` फ़ोल्डर या `--rnd-data` चेकआउट को पढ़ता है जिसे आप नाम देते हैं। यह rnd-data के निजी कंपेनियन और ब्लॉब स्टोर को अस्वीकार करता है, और यह केवल सापेक्ष पथ प्रिंट करता है।
- **यह क्या लिखता है:** केवल `--out` के अंतर्गत: इसका स्कैन कैश, एम्बेडिंग कैश, रिपोर्ट, `description-review.jsonl` और वर्कशीट। यह प्रोजेक्ट के अंदर `--out` को अस्वीकार करता है (`PERM_OUT_INSIDE_PROJECT`), इसलिए प्रोजेक्ट को कभी भी संशोधित नहीं किया जाता है।
- **यह क्या नहीं छूता:** उन दो फ़ोल्डरों के बाहर कुछ भी। यह कहीं भी कुछ भी नहीं हटाता है।
- **नेटवर्क:** लाइट टियर इसे कभी उपयोग नहीं करता है। एक मॉडल बैकएंड पहली बार उपयोग करने पर हगिंग फेस से अपने पिन किए गए भार प्राप्त करता है, जब तक कि `HF_HUB_OFFLINE=1` सेट न हो; ड्राफ्टर हर बार `HF_HUB_OFFLINE=1` और `local_files_only=True` के साथ लोड होता है। कोई भी मॉडल रिमोट कोड नहीं चलाता है।
- **अनुमतियाँ:** एक सामान्य उपयोगकर्ता खाता। एक GPU को `--device` और एक `--grant` आईडी की आवश्यकता होती है, इसलिए साझा कार्ड को गलती से नहीं लिया जाएगा। मॉडल लोड करने से पहले, एक VRAM गवर्नर जांचता है कि पहले से उपयोग में मौजूद मेमोरी और मॉडल की मापी गई आवश्यकता `--vram-budget` (डिफ़ॉल्ट 26,624 MiB) के अंतर्गत आती है। यह कभी भी कुछ भी हटाता नहीं है; `--wait` तब तक जांच करता रहता है जब तक कि लोड फिट न हो जाए।
- **गुप्त जानकारी और टेलीमेट्री:** यह कोई क्रेडेंशियल नहीं पढ़ता है, और यह कोई टेलीमेट्री एकत्र या भेजता नहीं है।
- **सुरक्षित चरण:** चल रहे बेंचमार्क के लिए लॉक किए गए प्रोजेक्ट पर, विवरण एम्बेडिंग और विवरणों की किसी भी मॉडल-आधारित स्वीकृति को तब तक अस्वीकार कर दिया जाता है जब तक कि `--allow-multimodal-pool` पास न हो जाए, ताकि बेंचमार्क के प्रश्नों को उन मॉडलों द्वारा आकार न दिया जाए जिन्हें यह मापता है।

[SECURITY.md](SECURITY.md) में वर्णित अनुसार किसी भेद्यता की रिपोर्ट करें।

## स्थिति

**v0.2.0, एक सार्वजनिक प्रोटोटाइप।** कोड ऑफ़लाइन परीक्षण किया गया: 188 परीक्षण और 95% लाइन कवरेज, मॉडल लाइब्रेरी के लिए छोटे स्टैंड-इन के साथ।
- **कैलिब्रेशन:** स्टूडियो अपने स्वयं के डेटा पर, एक RTX 5090 पर चलता है, डुप्लिकेट थ्रेशोल्ड और VRAM आंकड़े सेट करता है। वे अस्थायी हैं, और वे एक प्रकार के प्रति एक पूल पर निर्भर करते हैं। कोड थ्रेशोल्ड 17 प्लांटेड प्रतियों पर निर्भर करते हैं। विधि और डेटा [न्यूज़लेटर अंक 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/) में हैं।
- **बड़े प्रोजेक्ट:** डुप्लिकेट, बाहरी मान और लुक चैनल सीमित ब्लॉकों में काम करते हैं। सिंथेटिक सेट पर, निकट-डुप्लिकेट खोज और लुक चैनल 100,000 छवियों पर मिनटों में लगते हैं, जबकि पहले इसमें घंटों लगते थे या मेमोरी खत्म हो जाती थी। विवरण गेट लगभग 6 गुना तेज है। इसका जैकार्ड चेक अभी भी बहुत बड़े सेट पर तेजी से धीमा हो जाता है जो कई शब्दों को साझा करते हैं, और वह डिज़ाइन प्रश्न खुला है।
- **1.0 से पहले कमांड लाइन में बदलाव की अपेक्षा करें।**

डिज़ाइन रिकॉर्ड, इसका अनुसंधान आधार और इसका समीक्षा इतिहास स्टूडियो की अनुसंधान लाइब्रेरी में मौजूद है: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk)।

## विकास

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI लाइन कवरेज को 90% या उससे अधिक पर रखता है।

## लाइसेंस

[MIT](LICENSE)। इसमें जो मॉडल लोड किए जा सकते हैं, वे इस पैकेज का हिस्सा नहीं हैं, और प्रत्येक का अपना लाइसेंस है; उपयोग करने से पहले मॉडल कार्ड पढ़ें।

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
